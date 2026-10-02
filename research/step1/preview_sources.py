"""100条原始视频的中点实际帧总览，便于人工检查场景差异。"""
from common import ROOT, DATA
import json
import av
from PIL import Image, ImageDraw, ImageFont


def main():
    """按容器时间seek后逐帧检查PTS，展示实际命中的帧时刻。"""
    rows=json.loads((ROOT/"results/media_audit.json").read_text())
    canvas=Image.new("RGB",(1800,1600),"white")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",10)
    for i,row in enumerate(rows):
        v,c=row["sample_id"].split("$_$")
        with av.open(str(DATA/v/(c+".mp4"))) as container:
            stream=container.streams.video[0]
            target=row["duration"]*.5
            container.seek(int(target/float(stream.time_base)),stream=stream,backward=True)
            for frame in container.decode(video=0):
                if frame.time>=target:
                    break
            im=frame.to_image();im.thumbnail((178,130))
            x,y=(i%10)*180,(i//10)*160
            canvas.paste(im,(x,y))
            draw.text((x+2,y+132),row["sample_id"],font=font,fill="black")
            draw.text((x+2,y+146),f"PTS={frame.time:.3f}s  D={row['duration']:.3f}s",font=font,fill="black")
    target=ROOT/"results/figures/source_overview.jpg";target.parent.mkdir(exist_ok=True)
    canvas.save(target,quality=88)
    print(target)


if __name__=="__main__":
    main()
