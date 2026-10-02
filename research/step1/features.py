"""原视频解码与冻结的文本、声学、面部表情特征提取。"""
import numpy as np
import av
from PIL import Image


def decode_media(path, sampling_hz):
    """按媒体PTS解码，保留真实帧时刻；音频缺口保留为零并单独记录覆盖掩码。"""
    selected, all_pts, frame_indices = [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        rate = float(stream.average_rate)
        next_time = 0.
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise ValueError("视频帧缺少PTS，不能假造时间戳")
            t = float(frame.pts * frame.time_base) - origin
            all_pts.append(t)
            if t + 1e-8 >= next_time:
                im = frame.to_image()
                im.thumbnail((640, 640))
                selected.append((t, np.asarray(im.convert("RGB"))))
                frame_indices.append(index)
                next_time = (np.floor(t * sampling_hz) + 1) / sampling_hz
        duration = max(float(stream.duration * stream.time_base) if stream.duration else 0.,
                       all_pts[-1] + 1 / rate)
    pieces = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                pieces.append((float(out.pts * out.time_base) - origin,
                               out.to_ndarray().reshape(-1).copy()))
        for out in resampler.resample(None):
            pieces.append((float(out.pts * out.time_base) - origin,
                           out.to_ndarray().reshape(-1).copy()))
    if not pieces:
        raise ValueError("没有可解码音频")
    audio_start = min(t for t, _ in pieces)
    audio_end = max(t + len(x) / 16000 for t, x in pieces)
    waveform = np.zeros(int(round((audio_end - audio_start) * 16000)), np.float32)
    coverage = np.zeros(len(waveform), bool)
    for t, x in pieces:
        a = round((t - audio_start) * 16000)
        b = min(a + len(x), len(waveform))
        waveform[a:b] = x[:b-a]
        coverage[a:b] = True
    duration = max(duration, audio_end)
    pts = np.asarray([t for t, _ in selected])
    # 每个抽样帧代表相邻采样时刻中点之间的时间单元，属于显式近邻保持假设。
    edges = np.r_[0., (pts[:-1] + pts[1:]) / 2, duration]
    return dict(waveform=waveform, audio_coverage=coverage, audio_start=audio_start,
                duration=duration, frames=selected, video_pts=pts,
                video_bounds=np.column_stack([edges[:-1], edges[1:]]),
                video_frame_indices=np.asarray(frame_indices), all_video_pts=np.asarray(all_pts),
                video_rate=rate, video_origin=origin)


class Extractors:
    def __init__(self, assets, config, device):
        import torch
        import opensmile
        import mediapipe as mp
        from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC
        self.config, self.device, self.torch, self.mp = config, device, torch, mp
        self.tokenizer = AutoTokenizer.from_pretrained(assets / "distilbert", local_files_only=True)
        self.text_model = AutoModel.from_pretrained(assets / "distilbert", local_files_only=True,
                                                    use_safetensors=True).eval().to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(assets / "wav2vec2", local_files_only=True)
        self.ctc_model = Wav2Vec2ForCTC.from_pretrained(assets / "wav2vec2", local_files_only=True,
                                                      use_safetensors=True).eval().to(device)
        for model in (self.text_model, self.ctc_model):
            model.requires_grad_(False)
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                     num_workers=1)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(assets / "face_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=config["face_detection_threshold"],
            min_face_presence_confidence=config["face_presence_threshold"],
            output_face_blendshapes=True)
        self.face = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def acoustic(self, media):
        """读取openSMILE返回的窗口边界，区分静音、音频缺口与非有限数值。"""
        x = media["waveform"]
        df = self.smile.process_signal(x, 16000)
        starts = df.index.get_level_values("start").total_seconds().to_numpy()
        ends = df.index.get_level_values("end").total_seconds().to_numpy()
        bounds = np.column_stack([starts, ends]) + media["audio_start"]
        values = df.to_numpy(np.float32)
        a = np.clip(np.rint(starts * 16000).astype(int), 0, len(x))
        b = np.clip(np.rint(ends * 16000).astype(int), 0, len(x))
        cumulative = np.r_[0., np.cumsum(x.astype(float) ** 2)]
        rms = np.sqrt((cumulative[b] - cumulative[a]) / np.maximum(1, b-a))
        available = np.r_[0, np.cumsum(media["audio_coverage"])]
        observed = (available[b] - available[a] == b-a) & (b > a) & np.isfinite(values).all(axis=1)
        values[~np.isfinite(values)] = 0
        silence = rms < self.config["silence_rms_threshold"]
        return values, bounds, observed, rms.astype(np.float32), silence, list(df.columns)

    def visual(self, media):
        """逐帧检测；无人脸保留零向量及False掩码，多脸选择面积最大者并标记。"""
        features = np.zeros((len(media["frames"]), 52), np.float32)
        valid = np.zeros(len(features), bool)
        face_count = np.zeros(len(features), np.int16)
        names = None
        for i, (_, frame) in enumerate(media["frames"]):
            result = self.face.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                   data=np.ascontiguousarray(frame)))
            face_count[i] = len(result.face_landmarks)
            if not result.face_blendshapes:
                continue
            areas = []
            for points in result.face_landmarks:
                xs, ys = [p.x for p in points], [p.y for p in points]
                areas.append((max(xs)-min(xs)) * (max(ys)-min(ys)))
            index = int(np.argmax(areas))
            cats = result.face_blendshapes[index]
            this_names = [c.category_name for c in cats]
            if names is not None:
                assert names == this_names
            names = this_names
            features[i] = [c.score for c in cats]
            valid[i] = np.isfinite(features[i]).all()
        return features, valid, face_count, names

    def text(self, raw_text, words):
        """直接编码原始文本（含标点）；offset_mapping保留子词与原文词的精确关系。"""
        full = self.tokenizer(raw_text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offsets = full["input_ids"], full["offset_mapping"]
        embeddings, chunk_ids = [], []
        # 只在超过模型上限时分块，禁止静默截断任何原词或子词。
        chunk_size = self.config["text_max_subwords_per_chunk"]
        for chunk, start in enumerate(range(0, len(ids), chunk_size)):
            part = ids[start:start + chunk_size]
            inputs = self.torch.tensor([[self.tokenizer.cls_token_id] + part +
                                       [self.tokenizer.sep_token_id]], device=self.device)
            with self.torch.inference_mode():
                vectors = self.text_model(input_ids=inputs).last_hidden_state[0, 1:-1]
            embeddings.append(vectors.float().cpu().numpy())
            chunk_ids.extend([chunk] * len(part))
        subwords = np.concatenate(embeddings).astype(np.float32)
        word_features = np.zeros((len(words), subwords.shape[1]), np.float32)
        mask = np.zeros(len(words), bool)
        wordmap = np.full(len(ids), -1, np.int32)
        mapping = []
        for i, word in enumerate(words):
            js = [j for j, (a, b) in enumerate(offsets)
                  if min(b, word["char_end"]) > max(a, word["char_start"])]
            if js:
                word_features[i] = subwords[js].mean(axis=0)
                mask[i] = True
                wordmap[js] = i
            word["subword_indices"] = js
        for j, (token, (a, b)) in enumerate(zip(ids, offsets)):
            mapping.append(dict(index=j, token_id=token, token=self.tokenizer.convert_ids_to_tokens(token),
                                char_start=a, char_end=b, word_index=int(wordmap[j]),
                                chunk_index=chunk_ids[j]))
        return word_features, mask, subwords, np.asarray(ids), wordmap, mapping

    def emissions(self, media):
        """冻结声学模型，仅估计给定转写的时间；ASR文本只作诊断，不覆盖原文。"""
        x = media["waveform"]
        values = self.processor(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with self.torch.inference_mode():
            logits = self.ctc_model(values).logits[0]
            logp = logits.log_softmax(-1).float().cpu().numpy()
        decoded = self.processor.decode(logp.argmax(axis=-1))
        stride, receptive = 1, 1
        for kernel, step in zip(self.ctc_model.config.conv_kernel, self.ctc_model.config.conv_stride):
            receptive += (kernel - 1) * stride
            stride *= step
        centers = (np.arange(len(logp)) * stride + (receptive-1)/2) / 16000 + media["audio_start"]
        # 显式使用卷积感受野中心和步长，不用 t/T*视频长度近似。
        bounds = np.column_stack([centers-stride/32000, centers+stride/32000])
        bounds = np.clip(bounds, 0, media["duration"])
        return logp, bounds, decoded, dict(stride_samples=stride, receptive_field_samples=receptive)

    def close(self):
        self.face.close()


def contact_sheet(media, path):
    """保存等分位置的实际抽样帧，便于人工回看而非重新生成图片。"""
    from PIL import ImageDraw
    selected = np.linspace(0, len(media["frames"])-1, 6).round().astype(int)
    canvas = Image.new("RGB", (6*180, 155), "white")
    draw = ImageDraw.Draw(canvas)
    for col, index in enumerate(selected):
        t, frame = media["frames"][index]
        im = Image.fromarray(frame); im.thumbnail((178, 130))
        canvas.paste(im, (col*180, 0))
        draw.text((col*180+4, 135), f"PTS {t:.3f}s", fill="black")
    canvas.save(path, quality=85)
