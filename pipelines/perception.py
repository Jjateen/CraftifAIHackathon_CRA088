#!/usr/bin/env python3
# perception.py — YOLOv8 (vehicles) + MiDaS-small (depth) over a video, TensorRT FP16.
# Runs inside the PipeGen DeepStream container (polygraphy + TRT 10.8).
# Writes: annotated video (filesink) + results.json in the PipeGen frame schema
# consumed by adas_fusion. Perception only — FCW/BSD logic lives downstream.
#
# usage: python3 perception.py <in.mp4> <out_annotated.mp4> <out.json> \
#            <yolo.engine> <midas.engine>
import json, sys, time
import cv2
import numpy as np
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

COCO_VEH = {0:"person",1:"bicycle",2:"car",3:"motorcycle",5:"bus",7:"truck"}
CONF, NMS_IOU = 0.35, 0.45

def letterbox(img, th, tw):
    h, w = img.shape[:2]
    r = min(th/h, tw/w)
    nh, nw = int(h*r), int(w*r)
    out = np.full((th, tw, 3), 114, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh))
    return out, r

def nms(boxes, scores, thr):
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF, thr)
    return np.array(idx).reshape(-1)

def main():
    vin, vout, jout, yeng, meng = sys.argv[1:6]
    yolo  = TrtRunner(EngineFromBytes(open(yeng,  "rb").read()))
    midas = TrtRunner(EngineFromBytes(open(meng, "rb").read()))
    cap = cv2.VideoCapture(vin)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(vout, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frames, fi, t_ema = [], 0, 0.0

    with yolo, midas:
        yname = list(yolo.get_input_metadata().keys())[0]
        yshape = list(yolo.get_input_metadata()[yname].shape)    # [1,3,h,w]
        yh, yw = int(yshape[2]), int(yshape[3])
        mname = list(midas.get_input_metadata().keys())[0]
        mshape = list(midas.get_input_metadata()[mname].shape)   # [1,3,h,w]
        mh, mw = int(mshape[2]), int(mshape[3])
        while True:
            ok, img = cap.read()
            if not ok: break
            t0 = time.time()

            # ---- YOLOv8
            lb, r = letterbox(img, yh, yw)
            x = lb[:, :, ::-1].transpose(2,0,1)[None].astype(np.float32)/255.0
            pred = list(yolo.infer({yname: x}).values())[0]        # (1,84,8400)
            p = pred[0].T                                          # (8400,84)
            cls = p[:, 4:].argmax(1); score = p[:, 4:].max(1)
            keep = score > CONF
            p, cls, score = p[keep], cls[keep], score[keep]
            boxes = np.zeros((len(p), 4), np.float32)              # cx,cy,w,h -> x,y,w,h
            boxes[:,0] = (p[:,0] - p[:,2]/2)/r; boxes[:,1] = (p[:,1] - p[:,3]/2)/r
            boxes[:,2] = p[:,2]/r;              boxes[:,3] = p[:,3]/r

            # ---- MiDaS (relative inverse depth, higher = closer)
            mx = cv2.resize(img, (mw, mh))[:, :, ::-1].transpose(2,0,1)[None].astype(np.float32)
            mx = (mx/255.0 - np.array([0.485,0.456,0.406]).reshape(1,3,1,1)) / \
                 np.array([0.229,0.224,0.225]).reshape(1,3,1,1)
            dep = list(midas.infer({mname: mx.astype(np.float32)}).values())[0][0]
            if dep.ndim == 3: dep = dep[0]
            dep = cv2.resize(dep, (W, H))
            dmin, dmax = float(dep.min()), float(dep.max())
            dnorm = (dep - dmin) / max(dmax - dmin, 1e-6)          # 0..1, 1 = nearest

            objs = []
            if len(boxes):
                for i in nms(boxes, score, NMS_IOU):
                    c = int(cls[i])
                    if c not in COCO_VEH: continue
                    bx, by, bw, bh = [float(v) for v in boxes[i]]
                    bx, by = max(bx,0), max(by,0)
                    roi = dnorm[int(by):int(by+bh), int(bx):int(bx+bw)]
                    d = float(np.median(roi)) if roi.size else 0.0
                    objs.append({"label": COCO_VEH[c], "confidence": float(score[i]),
                                 "bbox": [bx, by, bw, bh], "depth": d})
                    cv2.rectangle(img, (int(bx),int(by)), (int(bx+bw),int(by+bh)), (80,200,255), 2)
                    cv2.putText(img, f"{COCO_VEH[c]} {score[i]:.2f} d={d:.2f}",
                                (int(bx), int(by)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,200,255), 2)

            dt = time.time() - t0
            t_ema = dt if not t_ema else 0.9*t_ema + 0.1*dt
            cv2.putText(img, f"pipeline {1.0/max(t_ema,1e-3):.0f} FPS (TRT FP16)",
                        (10, H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            wr.write(img)
            frames.append({"frame": fi, "source_id": 0,
                           "pts_ns": int(fi/fps*1e9), "width": W, "height": H,
                           "objects": objs, "shapes": []})
            fi += 1

    wr.release()
    json.dump({"pipeline": "video_json_export", "source": vin,
               "min_confidence": CONF, "frames": frames}, open(jout, "w"))
    print(f"done {fi} frames -> {vout}, {jout}")

if __name__ == "__main__":
    main()
