# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""CTC显式强制对齐、原文词索引与交叠聚合。"""
import re
import unicodedata
import numpy as np


def extract_words(text):
    """不改写原始转写；另存可发音词、原文字符区间和数字展开。"""
    from num2words import num2words
    words = []
    for m in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)*", text):
        original = m.group()
        norm = original.replace("’", "'")
        if norm[0].isdigit():
            value = norm.replace(",", "")
            norm = num2words(float(value) if "." in value else int(value), lang="en")
        norm = unicodedata.normalize("NFKD", norm).encode("ascii", "ignore").decode().upper()
        norm = " ".join(re.findall(r"[A-Z]+(?:'[A-Z]+)*", norm))
        words.append(dict(index=len(words), original=original, char_start=m.start(),
                          char_end=m.end(), normalized=norm))
    if not words:
        raise ValueError("原转写中没有可对齐的英文词")
    return words


def make_target(words, vocab):
    """把原文词展开为CTC字符，保留字符→原文词映射；词界符不算文字特征。"""
    chars, owners = [], []
    for i, word in enumerate(words):
        if i:
            chars.append("|"); owners.append(-1)
        for ch in word["normalized"].replace(" ", "|"):
            if ch not in vocab:
                raise ValueError(f"CTC词表不支持字符 {ch!r}")
            chars.append(ch); owners.append(i if ch != "|" else -1)
    return np.asarray([vocab[c] for c in chars], np.int32), np.asarray(owners), "".join(chars)


def ctc_viterbi(logp, targets, blank=0):
    """标准CTC最大概率单调路径；重复字符必须经blank，不采用均分时间伪对齐。"""
    T, _ = logp.shape
    labels = np.full(2 * len(targets) + 1, blank, np.int32)
    labels[1::2] = targets
    S = len(labels)
    if T < len(targets) + int(np.sum(targets[1:] == targets[:-1])):
        raise ValueError("CTC帧数不足以覆盖转写字符及重复字符")
    previous = np.full(S, -np.inf, np.float64)
    previous[0] = logp[0, blank]
    previous[1] = logp[0, labels[1]]
    pointers = np.zeros((T, S), np.uint8)
    can_skip = (np.arange(S) >= 2) & (labels != blank) & (labels != np.roll(labels, 2))
    for t in range(1, T):
        stay = previous
        advance = np.r_[-np.inf, previous[:-1]]
        skip = np.r_[-np.inf, -np.inf, previous[:-2]]
        skip[~can_skip] = -np.inf
        candidates = np.stack([stay, advance, skip])
        move = candidates.argmax(axis=0)
        pointers[t] = move
        previous = candidates[move, np.arange(S)] + logp[t, labels]
    state = S - 1 if previous[-1] >= previous[-2] else S - 2
    if not np.isfinite(previous[state]):
        raise ValueError("没有覆盖完整转写的有效CTC路径")
    score = float(previous[state])
    path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = state
        if t:
            state -= int(pointers[t, state])
    return path, score


def words_from_path(path, targets, owners, logp, words, frame_bounds, confidence_threshold):
    """仅使用发音字符占据的CTC帧确定词界；置信度是模型分数而非准确率。"""
    bounds = np.zeros((len(words), 2), np.float64)
    confidence = np.zeros(len(words), np.float32)
    valid = np.zeros(len(words), bool)
    states = (path - 1) // 2
    for i, word in enumerate(words):
        token_indices = np.flatnonzero(owners == i)
        frames = np.flatnonzero((path % 2 == 1) & np.isin(states, token_indices))
        if not len(frames):
            word.update(start=None, end=None, confidence=0., status="alignment_failed")
            continue
        bounds[i] = [frame_bounds[frames[0], 0], frame_bounds[frames[-1], 1]]
        assigned = targets[states[frames]]
        confidence[i] = np.exp(logp[frames, assigned].mean())
        valid[i] = True
        word.update(start=float(bounds[i, 0]), end=float(bounds[i, 1]),
                    confidence=float(confidence[i]),
                    status="low_confidence" if confidence[i] < confidence_threshold else "aligned")
    return bounds, confidence, valid


def overlap_pool(intervals, frame_bounds, features, observed):
    """按真实时间交叠长度归一化聚合；无有效观测时返回零和False掩码。"""
    overlaps = np.maximum(0., np.minimum(intervals[:, None, 1], frame_bounds[None, :, 1]) -
                          np.maximum(intervals[:, None, 0], frame_bounds[None, :, 0]))
    overlaps *= np.asarray(observed, bool)[None, :]
    denominator = overlaps.sum(axis=1)
    weights = np.divide(overlaps, denominator[:, None], out=np.zeros_like(overlaps),
                        where=denominator[:, None] > 0)
    pooled = (weights @ features).astype(np.float32)
    return pooled, denominator > 0, weights.astype(np.float32)


def window_bounds(duration, width):
    starts = np.arange(0., duration, width)
    return np.column_stack([starts, np.minimum(starts + width, duration)])


def word_error_rate(reference, hypothesis):
    """只作声学模型与给定转写的一致性诊断，不当作时间戳误差。"""
    a, b = reference.split(), hypothesis.split()
    previous = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        current = np.empty(len(b) + 1, int); current[0] = i
        for j, y in enumerate(b, 1):
            current[j] = min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y))
        previous = current
    return float(previous[-1] / max(1, len(a)))
