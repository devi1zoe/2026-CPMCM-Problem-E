"""独立于模型的100条原始媒体完整解码检查。"""
from common import ROOT, DATA, save_json
from features import decode_media
import csv
import numpy as np


def main():
    """完整遍历声画流，记录PTS、实际时长和解码失败，绝不只检查扩展名。"""
    rows=[]
    for path in sorted(DATA.rglob("*.mp4")):
        try:
            data=decode_media(path,5)
            rows.append(dict(sample_id=path.parent.name+"$_$"+path.stem,status="decoded",error="",
                             duration=data["duration"],audio_start=data["audio_start"],
                             audio_duration=len(data["waveform"])/16000,
                             audio_coverage=float(data["audio_coverage"].mean()),
                             video_frames=len(data["all_video_pts"]),
                             video_pts_monotonic=bool((np.diff(data["all_video_pts"])>0).all()),
                             audio_rms=float(np.sqrt(np.mean(data["waveform"].astype(float)**2)))))
        except Exception as e:
            rows.append(dict(sample_id=path.parent.name+"$_$"+path.stem,status="failed",error=repr(e)))
    output=ROOT/"results";output.mkdir(exist_ok=True)
    save_json(output/"media_audit.json",rows)
    with open(output/"media_audit.csv","w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(dict(total=len(rows),failed=sum(r["status"]!="decoded" for r in rows)))


if __name__=="__main__":
    main()
