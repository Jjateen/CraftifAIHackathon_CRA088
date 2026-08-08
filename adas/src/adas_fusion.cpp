// adas_fusion — L2 ADAS decision layer over PipeGen perception JSON.
//
// Reads the per-frame detections / lane polylines / depth that a PipeGen
// DeepStream pipeline wrote (results.json) alongside the source video, runs
// the four ADAS functions, and writes an annotated demo video.
//
//   front camera: FCW (YOLOv8 objects + MiDaS depth)
//   rear  camera: BSD (YOLOv8 objects + MiDaS depth + zones)
//
// Algorithm shapes and thresholds follow the ADAS_modular v3.3 reference
// (assets/{fcw,bsd}/[*]_config.yaml, src/features/*), simplified to a
// single-file monocular implementation:
//   FCW  : TTC ladder 2.3 / 2.8 / 3.2 / 3.6 s, cooldown 2 s, max range 40 m
//   BSD  : critical 2.5 m x 5 m, overtake 8 m x 5 m zones, N-frame persistence
//
// Usage:
//   adas_fusion --mode front --video front.mov --json results_front.json --out demo_front.mp4
//   adas_fusion --mode rear  --video rear.mov  --json results_rear.json  --out demo_rear.mp4

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------- params ---
struct Params {                       // v3.3-derived defaults
    // FCW
    double fcw_ttc_warn      = 3.2;   // fcw_ttc_default
    double fcw_ttc_critical  = 2.3;   // fcw_ttc_aggressive
    double fcw_max_dist_m    = 40.0;  // fcw_max_distance
    double fcw_cooldown_s    = 2.0;
    // LDW
    double ldw_ttd_s         = 3.0;   // ldw_ttd_threshold_default
    double ldw_cooldown_s    = 7.0;
    // ACC
    double acc_headway_min_s = 0.6;   // min_headway_time
    double acc_reaction_s    = 0.18;  // headway_reaction_time
    double ego_speed_mps     = 16.7;  // assumed cruise 60 km/h (no CAN feed)
    // BSD
    double bsd_critical_l_m  = 2.5;   // critical_l
    double bsd_overtake_l_m  = 8.0;   // overtake_l
    int    bsd_persist       = 10;    // warning_threshold (frames)
    // monocular distance fallback: dist = f_px * H_car / bbox_h
    double cam_f_px_per_w    = 0.9;   // f ~ 0.9 * frame width
    double veh_height_m      = 1.5;
};

// ------------------------------------------------------------- json load ---
struct Obj { cv::Rect2d box; std::string label; double conf{}, depth{-1}; };
struct Frame {
    int idx{}; double pts_s{};
    std::vector<Obj> objs;
    std::vector<std::vector<cv::Point2f>> lanes;
};

// The pipeline may be stopped mid-write, leaving a truncated array; parse
// leniently and keep whatever frames are intact.
static std::vector<Frame> loadFrames(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; exit(1); }
    std::string s((std::istreambuf_iterator<char>(f)), {});
    json j = json::parse(s, nullptr, false);
    if (j.is_discarded()) {                         // truncated: cut at last '}'
        auto pos = s.rfind("},");
        while (pos != std::string::npos) {
            std::string fixed = s.substr(0, pos + 1) + "]}";
            j = json::parse(fixed, nullptr, false);
            if (!j.is_discarded()) break;
            pos = s.rfind("},", pos - 1);
        }
        if (j.is_discarded()) { std::cerr << "unparseable json\n"; exit(1); }
        std::cerr << "warning: truncated json recovered\n";
    }
    std::vector<Frame> out;
    for (auto& jf : j["frames"]) {
        Frame fr;
        fr.idx   = jf.value("frame", (int)out.size());
        fr.pts_s = jf.value("pts_ns", 0.0) / 1e9;
        for (auto& jo : jf.value("objects", json::array())) {
            Obj o;
            o.label = jo.value("label", jo.value("class", std::string("obj")));
            o.conf  = jo.value("confidence", jo.value("conf", 0.0));
            o.depth = jo.value("depth", -1.0);
            if (jo.contains("bbox")) {              // [x, y, w, h]
                auto b = jo["bbox"];
                o.box = {b[0].get<double>(), b[1].get<double>(),
                         b[2].get<double>(), b[3].get<double>()};
            } else {
                o.box = {jo.value("left", 0.0), jo.value("top", 0.0),
                         jo.value("width", 0.0), jo.value("height", 0.0)};
            }
            fr.objs.push_back(std::move(o));
        }
        for (auto& js : jf.value("shapes", json::array())) {
            if (js.value("kind", "") != "polyline") continue;
            std::vector<cv::Point2f> pts;
            for (auto& p : js["points"])
                pts.emplace_back(p[0].get<float>(), p[1].get<float>());
            if (pts.size() > 1) fr.lanes.push_back(std::move(pts));
        }
        out.push_back(std::move(fr));
    }
    return out;
}

// ------------------------------------------------------------- utilities ---
static bool isVehicle(const std::string& l) {
    static const char* v[] = {"car","truck","bus","motorcycle","bicycle","vehicle","van"};
    for (auto* s : v) if (l.find(s) != std::string::npos) return true;
    return false;
}

// Monocular range fusion: bbox pinhole gives absolute scale, MiDaS relative
// inverse depth gives cross-object structure. Per frame, fit the scale
// s = median(pinhole_i * midas_i) over detected vehicles, then
// range_i = s / midas_i. Falls back to pure pinhole when depth is absent.
static double pinholeMeters(const Obj& o, int W, const Params& P) {
    double f = P.cam_f_px_per_w * W;
    return o.box.height > 1 ? f * P.veh_height_m / o.box.height : 1e9;
}
static std::vector<double> fuseRanges(const std::vector<Obj>& objs, int W, const Params& P) {
    std::vector<double> pin(objs.size()), out(objs.size());
    std::vector<double> scales;
    for (size_t i = 0; i < objs.size(); ++i) {
        pin[i] = pinholeMeters(objs[i], W, P);
        if (objs[i].depth > 0.02 && pin[i] < 500)
            scales.push_back(pin[i] * objs[i].depth);
    }
    double s = -1;
    if (!scales.empty()) {
        std::nth_element(scales.begin(), scales.begin() + scales.size()/2, scales.end());
        s = scales[scales.size()/2];
    }
    for (size_t i = 0; i < objs.size(); ++i)
        out[i] = (s > 0 && objs[i].depth > 0.02) ? s / objs[i].depth : pin[i];
    return out;
}

struct Banner { std::string txt; cv::Scalar color; };

static void drawBanners(cv::Mat& img, const std::vector<Banner>& bs) {
    int y = 12;
    for (auto& b : bs) {
        cv::Size ts = cv::getTextSize(b.txt, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr);
        cv::rectangle(img, {10, y, ts.width + 20, ts.height + 18}, b.color, cv::FILLED);
        cv::putText(img, b.txt, {20, y + ts.height + 8},
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, {255,255,255}, 2);
        y += ts.height + 26;
    }
}

// ------------------------------------------------------------------ main ---
int main(int argc, char** argv) {
    std::map<std::string,std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i+1];
    if (!a.count("--mode") || !a.count("--video") || !a.count("--json") || !a.count("--out")) {
        std::cerr << "usage: adas_fusion --mode front|rear --video in --json results.json --out demo.mp4\n";
        return 1;
    }
    const bool front = a["--mode"] == "front";
    Params P;
    if (a.count("--ego-kmh")) P.ego_speed_mps = std::stod(a["--ego-kmh"]) / 3.6;

    auto frames = loadFrames(a["--json"]);
    std::map<int,const Frame*> byIdx;
    for (auto& f : frames) byIdx[f.idx] = &f;

    cv::VideoCapture cap(a["--video"]);
    if (!cap.isOpened()) { std::cerr << "cannot open video\n"; return 1; }
    double fps = cap.get(cv::CAP_PROP_FPS);  if (fps <= 1) fps = 25;
    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter out(a["--out"], cv::VideoWriter::fourcc('m','p','4','v'), fps, {W, H});

    // feature state
    double lastFcw = -1e9, lastLdw = -1e9;
    std::deque<std::pair<double,double>> leadHist;   // t, range
    int bsdCountL = 0, bsdCountR = 0;
    double emaProcFps = 0;

    // BSD zones (rear camera image plane): bottom corners, sized so that the
    // zone's far edge ~ overtake_l with the pinhole model above.
    std::vector<cv::Point> zoneL = {{0,H},{(int)(0.38*W),H},{(int)(0.28*W),(int)(0.62*H)},{0,(int)(0.62*H)}};
    std::vector<cv::Point> zoneR = {{W,H},{(int)(0.62*W),H},{(int)(0.72*W),(int)(0.62*H)},{W,(int)(0.62*H)}};

    cv::Mat img;
    for (int fi = 0; cap.read(img); ++fi) {
        auto t0 = std::chrono::steady_clock::now();
        double t = fi / fps;
        std::vector<Banner> banners;
        const Frame* fr = byIdx.count(fi) ? byIdx[fi] : nullptr;

        if (fr) {
            // ---- draw objects, find lead vehicle / zone occupancy
            const Obj* lead = nullptr; double leadRange = 1e9;
            auto ranges = fuseRanges(fr->objs, W, P);
            for (size_t oi = 0; oi < fr->objs.size(); ++oi) {
                const auto& o = fr->objs[oi];
                double r = ranges[oi];
                cv::Scalar c = isVehicle(o.label) ? cv::Scalar(80,200,255) : cv::Scalar(180,180,180);
                cv::rectangle(img, o.box, c, 2);
                char lbl[96];
                snprintf(lbl, sizeof lbl, "%s %.0fm", o.label.c_str(), r < 500 ? r : 0.0);
                cv::putText(img, lbl, o.box.tl() + cv::Point2d(0,-6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, c, 2);
                if (!isVehicle(o.label)) continue;

                if (front) {
                    // lead: vehicle whose bottom-center sits in the ego corridor
                    double cx = o.box.x + o.box.width / 2;
                    if (cx > 0.35*W && cx < 0.65*W && r < leadRange && r < P.fcw_max_dist_m)
                        { lead = &o; leadRange = r; }
                } else {
                    cv::Point bc((int)(o.box.x + o.box.width/2), (int)(o.box.y + o.box.height));
                    if (r < P.bsd_overtake_l_m + 6) {   // ignore far background
                        if (cv::pointPolygonTest(zoneL, bc, false) >= 0) bsdCountL++;
                        else if (cv::pointPolygonTest(zoneR, bc, false) >= 0) bsdCountR++;
                    }
                }
            }

            if (front) {
                // ---- FCW + ACC on the lead vehicle
                if (lead) {
                    leadHist.emplace_back(t, leadRange);
                    while (leadHist.size() > 12) leadHist.pop_front();
                    double ttc = 1e9, closing = 0;
                    if (leadHist.size() >= 4) {
                        double dt = leadHist.back().first  - leadHist.front().first;
                        double dr = leadHist.front().second - leadHist.back().second;
                        if (dt > 0.01) closing = dr / dt;                // m/s toward us
                        if (closing > 0.3) ttc = leadRange / closing;
                    }
                    cv::rectangle(img, lead->box, cv::Scalar(0,0,255), 3);
                    if (ttc < P.fcw_ttc_critical && t - lastFcw > P.fcw_cooldown_s) {
                        banners.push_back({"FCW: BRAKE! TTC " + cv::format("%.1fs", ttc), {0,0,200}});
                        lastFcw = t;
                    } else if (ttc < P.fcw_ttc_warn) {
                        banners.push_back({"FCW: warning, TTC " + cv::format("%.1fs", ttc), {0,90,230}});
                    }
                    // headway readout (informational; FCW is the actor)
                    double headway = leadRange / std::max(P.ego_speed_mps, 0.1);
                    cv::putText(img, cv::format("lead %.0fm  headway %.1fs", leadRange, headway),
                                {10, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {200,200,200}, 2);
                } else {
                    leadHist.clear();
                }
            } else {
                // ---- BSD persistence + zones
                cv::polylines(img, zoneL, true, {0,140,255}, 2);
                cv::polylines(img, zoneR, true, {0,140,255}, 2);
                bsdCountL = std::min(bsdCountL, 3*P.bsd_persist);   // clamp
                bsdCountR = std::min(bsdCountR, 3*P.bsd_persist);
                if (bsdCountL >= P.bsd_persist)
                    banners.push_back({"BSD: vehicle in LEFT blind spot", {0,80,255}});
                if (bsdCountR >= P.bsd_persist)
                    banners.push_back({"BSD: vehicle in RIGHT blind spot", {0,80,255}});
                // decay one per frame with no hit (simple leaky counter)
                bsdCountL = std::max(0, bsdCountL - 1);
                bsdCountR = std::max(0, bsdCountR - 1);
            }
        }

        // ---- metrics overlay (proc FPS = this loop, incl. decode+draw+encode)
        double ms = std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        double inst = ms > 0 ? 1000.0 / ms : 0;
        emaProcFps = emaProcFps == 0 ? inst : 0.9*emaProcFps + 0.1*inst;
        cv::putText(img, cv::format("proc %.0f FPS | video %.0f FPS | frame %d",
                    emaProcFps, fps, fi), {10, H-14},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,255}, 2);
        drawBanners(img, banners);
        out.write(img);
    }
    std::cout << "wrote " << a["--out"] << "\n";
    return 0;
}
