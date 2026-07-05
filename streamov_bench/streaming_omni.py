import hashlib
import json
import os
import os.path as osp
import re
import subprocess
import warnings

import numpy as np
import pandas as pd

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'
STREAMING_OMNI_JSONL = (
    'your_path_to'
    'streamov_bench.jsonl'
)


def _safe_str(x) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and pd.isna(x):
        return ''
    return str(x)


def _load_json_str(s, default):
    s = _safe_str(s)
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _extract_choice_letter(pred: str) -> str:
    s = _safe_str(pred).strip()
    if not s or FAIL_MSG in s:
        return ''

    try:
        answers = re.findall(r'<\s*answer\s*>(.*?)<\s*/\s*answer\s*>', s, flags=re.IGNORECASE | re.DOTALL)
        if answers:
            s = _safe_str(answers[-1]).strip()
    except Exception:
        pass

    m_json = re.search(r'"answer"\s*:\s*"([ABCDE])"', s, flags=re.IGNORECASE)
    if m_json:
        return m_json.group(1).upper()

    boxed_patterns = [
        r'\\boxed\s*\{\s*([ABCDE])\s*\}',
        r'\\box\s*\{\s*([ABCDE])\s*\}',
        r'\*+\s*([ABCDE])\s*\*+',
    ]
    for pat in boxed_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    s_up = s.upper()
    for prefix in [
        'THE ANSWER IS',
        'ANSWER IS',
        'THE CORRECT ANSWER IS',
        'THE CORRECT OPTION IS',
        'THE BEST ANSWER IS',
        'BEST ANSWER',
        'BEST OPTION',
        'ANSWER',
        'OPTION',
    ]:
        s_up = s_up.replace(prefix, ' ')

    patterns = [
        r'\(([ABCDE])\)',
        r'\b([ABCDE])\b',
        r'^\s*([ABCDE])\s*[\.\)]',
    ]
    for pat in patterns:
        m = re.search(pat, s_up)
        if m:
            return m.group(1)

    m = re.search(r'[ABCDE]', s_up)
    return m.group(0) if m else ''


def _acc(scores: list[int]) -> float:
    valid = [s for s in scores if s >= 0]
    if not valid:
        return float('nan')
    return float(np.mean(valid))


def _choice_index_to_letter(idx) -> str:
    try:
        idx = int(idx)
    except Exception:
        return ''
    if idx < 0 or idx >= 26:
        return ''
    return chr(ord('A') + idx)


def _entry_meta_turn_count(meta: dict) -> int:
    if 'correct_indices' in meta:
        return len(meta['correct_indices'])
    if 'correct_index' in meta:
        return 1
    return 0


def _parse_user_content(content: str) -> tuple[str, list[str]]:
    content = _safe_str(content).strip()
    content = re.sub(r'^\s*<video>\s*', '', content, flags=re.IGNORECASE)
    head, sep, tail = content.partition('Choices:')
    question = head.strip()
    choices_text = tail if sep else ''
    choices = []
    for line in choices_text.splitlines():
        line = line.strip()
        if re.match(r'^\(?[A-Z]\)?[\.\)]', line):
            choices.append(line)
    return question, choices


class StreamingOmni(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['exact_matching']
    MODALITY = 'VIDEO'
    STREAMING_SESSION_EVAL = True

    SYS = (
        'You are a helpful assistant for streaming video understanding. The media clips are ordered '
        'chronologically from earlier to later in the same session. Use the accumulated visual/audio '
        'context and the dialogue history to answer the current multiple-choice question.'
    )

    QUESTION_TMPL = (
        'Current turn type: {turn_type}\n'
        '{question_prefix}'
        'Question: {question}\n'
        'Choices:\n'
        '{choices}\n'
        'Respond with only the option letter (A-E) on the final line. '
        'Do not include explanation, markdown emphasis, or LaTeX boxes.'
    )

    HISTORY_TMPL = (
        'Dialogue history for relevant previous turns:\n'
        '{history}\n'
        'End of history.'
    )

    def __init__(
        self,
        dataset='Streaming-Omni',
        mode: str = 'sample_level_multi_turn',
        use_audio: bool = False,
        pack: bool = False,
        nframe: int = 0,
        fps: float = -1,
    ):
        self.mode = mode
        self.use_audio = use_audio
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        self.dataset_name = dataset
        self.audio_cache_root = osp.join(self.data_root, 'streaming_omni_audio')
        self.concat_cache_root = osp.join(self.data_root, 'streaming_omni_concat')
        os.makedirs(self.audio_cache_root, exist_ok=True)
        os.makedirs(self.concat_cache_root, exist_ok=True)

    @classmethod
    def supported_datasets(cls):
        return ['Streaming-Omni', 'Streaming_Omni', 'streaming_omni', 'StreamingOmni']

    def prepare_dataset(self, dataset):
        jsonl_path = os.environ.get('STREAMING_OMNI_JSONL', STREAMING_OMNI_JSONL)
        if not osp.exists(jsonl_path):
            raise FileNotFoundError(f'Streaming-Omni JSONL not found: {jsonl_path}')

        root = osp.dirname(jsonl_path)
        # One fixed TSV name would silently reuse stale rows when the JSONL path
        # changes but stays in the same directory; key the cache file by jsonl path.
        jsonl_tag = hashlib.md5(osp.abspath(jsonl_path).encode('utf-8')).hexdigest()[:16]
        data_file = osp.join(root, f'Streaming-Omni_{jsonl_tag}.tsv')

        def generate_tsv():
            rows = []
            total_assistant_turns = 0
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for session_id, raw_line in enumerate(f):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    item = json.loads(raw_line)
                    messages = item.get('messages', [])
                    videos = item.get('videos', [])
                    entry_metadata = item.get('entry_metadata', [])

                    try:
                        if len(messages) % 2 != 0:
                            raise ValueError(f'Session {session_id} has odd number of messages.')

                        user_turns = []
                        assistant_turns = []
                        turn_types = []
                        for i in range(0, len(messages), 2):
                            user_msg = messages[i]
                            assistant_msg = messages[i + 1]
                            if user_msg.get('role') != 'user' or assistant_msg.get('role') != 'assistant':
                                raise ValueError(
                                    f'Session {session_id} has non user/assistant alternating messages.'
                                )
                            user_turns.append(_safe_str(user_msg.get('content')))
                            assistant_turns.append(_safe_str(assistant_msg.get('content')))
                            turn_types.append(_safe_str(assistant_msg.get('category')))

                        if len(videos) > len(user_turns):
                            raise ValueError(
                                f'Session {session_id} has {len(videos)} videos but only {len(user_turns)} user turns.'
                            )

                        # Resolve "stem" question per turn (carry forward when user message has no new stem).
                        resolved_questions = []
                        current_resolved_question = ''
                        for user_content in user_turns:
                            question, _choices = _parse_user_content(user_content)
                            if question:
                                current_resolved_question = question
                            resolved_questions.append(current_resolved_question)

                        # Group boundaries must follow entry_metadata: some turns only have "<video>\\n\\nChoices:"
                        # and must start a new group even though _parse_user_content yields an empty stem question.
                        groups = []
                        turn_ptr = 0
                        for group_id, meta in enumerate(entry_metadata):
                            n_turns = _entry_meta_turn_count(meta)
                            if n_turns <= 0:
                                raise ValueError(
                                    f'Session {session_id} group {group_id} entry_metadata missing '
                                    f'correct_index/correct_indices.'
                                )
                            if turn_ptr + n_turns > len(user_turns):
                                raise ValueError(
                                    f'Session {session_id} group {group_id} exceeds session length: '
                                    f'need turns up to {turn_ptr + n_turns - 1}, have {len(user_turns)}.'
                                )
                            turn_indices = list(range(turn_ptr, turn_ptr + n_turns))
                            groups.append(
                                dict(
                                    group_id=group_id,
                                    start_turn=turn_indices[0],
                                    turn_indices=turn_indices,
                                )
                            )
                            turn_ptr += n_turns
                        if turn_ptr != len(user_turns):
                            raise ValueError(
                                f'Session {session_id}: entry_metadata covers {turn_ptr} turns but '
                                f'session has {len(user_turns)} user turns.'
                            )

                        turn_to_group = {}
                        group_sizes = {}
                        for group, meta in zip(groups, entry_metadata):
                            if 'correct_indices' in meta:
                                expected = list(meta['correct_indices'])
                            elif 'correct_index' in meta:
                                expected = [meta['correct_index']]
                            else:
                                raise ValueError(
                                    f'Session {session_id} group {group["group_id"]} missing answer metadata.'
                                )

                            if len(expected) != len(group['turn_indices']):
                                raise ValueError(
                                    f'Session {session_id} group {group["group_id"]} has '
                                    f'{len(group["turn_indices"])} turns but metadata has {len(expected)} answers.'
                                )

                            group_sizes[group['group_id']] = len(group['turn_indices'])
                            for step_id, (turn_idx, answer_idx) in enumerate(zip(group['turn_indices'], expected)):
                                turn_to_group[turn_idx] = {
                                    'group_id': group['group_id'],
                                    'group_start_turn_id': group['start_turn'],
                                    'step_id': step_id,
                                    'answer_by_meta': _choice_index_to_letter(answer_idx),
                                }

                        for turn_idx, (user_content, assistant_content, turn_type) in enumerate(
                            zip(user_turns, assistant_turns, turn_types)
                        ):
                            raw_question, choices = _parse_user_content(user_content)
                            resolved_question = resolved_questions[turn_idx]
                            group_info = turn_to_group[turn_idx]
                            video_path = _safe_str(videos[turn_idx]) if turn_idx < len(videos) else ''
                            answer_from_assistant = _extract_choice_letter(assistant_content)
                            answer_from_meta = group_info['answer_by_meta']
                            if (
                                answer_from_assistant
                                and answer_from_meta
                                and answer_from_assistant != answer_from_meta
                            ):
                                raise ValueError(
                                    f'Session {session_id} turn {turn_idx} answer mismatch: '
                                    f'{answer_from_assistant} vs {answer_from_meta}'
                                )
                            answer = answer_from_assistant or answer_from_meta

                            rows.append(
                                dict(
                                    index=len(rows),
                                    video=f'streaming_session_{session_id}',
                                    question=resolved_question,
                                    answer=answer,
                                    candidates=json.dumps(choices, ensure_ascii=False),
                                    session_id=session_id,
                                    session_turn_id=turn_idx,
                                    session_num_turns=len(user_turns),
                                    group_id=group_info['group_id'],
                                    group_start_turn_id=group_info['group_start_turn_id'],
                                    step_id=group_info['step_id'],
                                    group_num_turns=group_sizes[group_info['group_id']],
                                    turn_type=turn_type or 'Unknown',
                                    current_video_path=_safe_str(video_path),
                                    current_user_content=json.dumps(user_content, ensure_ascii=False),
                                    current_assistant_content=json.dumps(assistant_content, ensure_ascii=False),
                                    session_video_paths=json.dumps(videos, ensure_ascii=False),
                                    session_user_contents=json.dumps(user_turns, ensure_ascii=False),
                                    session_assistant_contents=json.dumps(assistant_turns, ensure_ascii=False),
                                )
                            )

                        total_assistant_turns += len(assistant_turns)
                    except ValueError as exc:
                        warnings.warn(f'Streaming-Omni JSONL session {session_id} skipped: {exc}', stacklevel=2)
                        continue

            df = pd.DataFrame(rows)
            if len(df) != total_assistant_turns:
                raise ValueError(
                    f'Expanded rows {len(df)} do not match assistant turns {total_assistant_turns}.'
                )
            df.to_csv(data_file, sep='\t', index=False)

        if not osp.exists(data_file):
            generate_tsv()

        return dict(root=root, data_file=data_file)

    def _audio_path(self, video_path: str) -> str:
        video_path = _safe_str(video_path)
        if not video_path:
            return ''

        sibling_wav = osp.splitext(video_path)[0] + '.wav'
        if osp.exists(sibling_wav):
            return sibling_wav

        cache_name = hashlib.md5(video_path.encode('utf-8')).hexdigest() + '.wav'
        cache_path = osp.join(self.audio_cache_root, cache_name)
        if osp.exists(cache_path):
            return cache_path

        try:
            from moviepy.editor import VideoFileClip

            clip = VideoFileClip(video_path)
            if clip.audio is None:
                clip.close()
                return ''
            clip.audio.write_audiofile(cache_path, verbose=False, logger=None)
            clip.audio.close()
            clip.close()
            return cache_path if osp.exists(cache_path) else ''
        except Exception:
            return ''

    def _context_cache_key(self, video_paths: list[str]) -> str:
        normed = [osp.abspath(_safe_str(p)) for p in video_paths if _safe_str(p)]
        joined = '\n'.join(normed)
        return hashlib.md5(joined.encode('utf-8')).hexdigest()

    def _concat_list_path(self, cache_key: str) -> str:
        return osp.join(self.concat_cache_root, f'{cache_key}.txt')

    def _merged_video_path(self, cache_key: str) -> str:
        return osp.join(self.concat_cache_root, f'{cache_key}.mp4')

    def _merged_audio_path(self, cache_key: str) -> str:
        return osp.join(self.audio_cache_root, f'{cache_key}.wav')

    def _ffmpeg_concat_escape(self, path: str) -> str:
        return path.replace("'", r"'\''")

    def _build_merged_context_media(self, video_paths: list[str], need_audio: bool = False) -> tuple[str, str]:
        video_paths = [_safe_str(p) for p in video_paths if _safe_str(p)]
        if not video_paths:
            return '', ''
        if len(video_paths) == 1:
            single_video = video_paths[0]
            single_audio = self._audio_path(single_video) if need_audio else ''
            return single_video, single_audio

        cache_key = self._context_cache_key(video_paths)
        merged_video = self._merged_video_path(cache_key)
        merged_audio = self._merged_audio_path(cache_key) if need_audio else ''

        if osp.exists(merged_video) and (not need_audio or osp.exists(merged_audio)):
            return merged_video, merged_audio

        list_file = self._concat_list_path(cache_key)
        lock_path = osp.join(self.concat_cache_root, f'{cache_key}.lock')
        with portalocker.Lock(lock_path, 'w', timeout=60):
            if not osp.exists(list_file):
                with open(list_file, 'w', encoding='utf-8') as f:
                    for p in video_paths:
                        abs_p = osp.abspath(p)
                        f.write(f"file '{self._ffmpeg_concat_escape(abs_p)}'\n")

            if not osp.exists(merged_video):
                # Prefer stream copy for speed; fall back to re-encode if concat copy fails.
                copy_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', list_file,
                    '-c', 'copy',
                    merged_video,
                ]
                ret = subprocess.run(copy_cmd, capture_output=True, text=True)
                if ret.returncode != 0:
                    recode_cmd = [
                        'ffmpeg', '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', list_file,
                        '-c:v', 'libx264',
                        '-preset', 'veryfast',
                        '-crf', '18',
                        '-c:a', 'aac',
                        merged_video,
                    ]
                    ret = subprocess.run(recode_cmd, capture_output=True, text=True)
                    if ret.returncode != 0:
                        raise RuntimeError(
                            f'Failed to concatenate streaming clips into {merged_video}: {ret.stderr}'
                        )

            if need_audio and not osp.exists(merged_audio):
                audio_cmd = [
                    'ffmpeg', '-y',
                    '-i', merged_video,
                    '-vn',
                    '-ac', '1',
                    '-ar', '16000',
                    merged_audio,
                ]
                ret = subprocess.run(audio_cmd, capture_output=True, text=True)
                if ret.returncode != 0:
                    warnings.warn(
                        f'Failed to extract merged audio for {merged_video}: {ret.stderr[:300]}'
                    )
                    merged_audio = ''

        if need_audio and merged_audio and not osp.exists(merged_audio):
            merged_audio = ''
        return merged_video, merged_audio

    def _clip_key(self, clip_path: str) -> str:
        clip_path = _safe_str(clip_path)
        stem = osp.splitext(osp.basename(clip_path))[0]
        short = hashlib.md5(clip_path.encode('utf-8')).hexdigest()[:10]
        return f'{stem}-{short}'

    def _frame_paths_for_clip(self, clip_key: str, num_frames: int):
        frame_root = osp.join(self.frame_root, clip_key)
        os.makedirs(frame_root, exist_ok=True)
        if self.fps > 0:
            return [
                osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, self.fps))
                for i in range(1, num_frames + 1)
            ]
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def _save_clip_frames(self, clip_path: str):
        import decord

        clip_key = self._clip_key(clip_path)
        vid = decord.VideoReader(clip_path)
        video_info = {'fps': float(vid.get_avg_fps()), 'n_frames': int(len(vid))}

        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self._frame_paths_for_clip(clip_key, self.nframe)
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / max(video_info['fps'], 1e-6)
            required_frames = max(int(total_duration * self.fps), 1)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self._frame_paths_for_clip(clip_key, len(indices))
        else:
            raise ValueError('nframe or fps must be set for Streaming-Omni')

        if not np.all([osp.exists(p) for p in frame_paths]):
            lock_path = osp.join(self.frame_root, clip_key + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = []
                    for i in indices:
                        i = min(max(i, 0), max(len(vid) - 1, 0))
                        images.append(vid[i].asnumpy())
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)
        return frame_paths

    def _context_video_paths(self, line) -> list[str]:
        session_videos = _load_json_str(line.get('session_video_paths', '[]'), [])
        cur_turn = int(line.get('session_turn_id', 0))
        return [_safe_str(v) for v in session_videos[: min(cur_turn + 1, len(session_videos))]]

    def _history_turn_indices(self, line) -> list[int]:
        cur_turn = int(line.get('session_turn_id', 0))
        if self.mode == 'sample_level_multi_turn':
            start_turn = 0
        elif self.mode == 'group_level_multi_turn':
            start_turn = int(line.get('group_start_turn_id', 0))
        else:
            raise ValueError(f'Unknown Streaming-Omni mode: {self.mode}')
        return list(range(start_turn, cur_turn))

    def _resolved_question_for_turn(self, user_contents: list[str], turn_idx: int) -> str:
        resolved = ''
        for i in range(turn_idx + 1):
            question, _ = _parse_user_content(user_contents[i])
            if question:
                resolved = question
        return resolved

    def _history_text(self, line) -> str:
        user_contents = _load_json_str(line.get('session_user_contents', '[]'), [])
        assistant_contents = _load_json_str(line.get('session_pred_contents', ''), None)
        if assistant_contents is None:
            assistant_contents = [''] * len(user_contents)
        history_lines = []
        for turn_idx in self._history_turn_indices(line):
            _, choices = _parse_user_content(user_contents[turn_idx])
            question = self._resolved_question_for_turn(user_contents, turn_idx)
            choice_str = '\n'.join(choices)
            block = (
                f'Turn {turn_idx + 1}\n'
                f'User question: {question}\n'
                f'Choices:\n{choice_str}'
            )
            pred = _safe_str(assistant_contents[turn_idx]).strip() if turn_idx < len(assistant_contents) else ''
            if pred:
                block += f'\nAssistant answer: {pred}'
            history_lines.append(block)
        return '\n\n'.join(history_lines)

    def build_prompt(self, line, video_llm: bool):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        message = [dict(type='text', value=self.SYS)]
        context_videos = self._context_video_paths(line)
        merged_video_path, merged_audio_path = self._build_merged_context_media(
            context_videos,
            need_audio=self.use_audio,
        )
        turn_type = _safe_str(line.get('turn_type', 'Unknown'))
        mode_label = 'entire session history' if self.mode == 'sample_level_multi_turn' else 'current group history'

        message.append(
            dict(
                type='text',
                value=(
                    f'The following media clips are ordered chronologically for one streaming session. '
                    f'Use all clips as context. Dialogue accumulation mode: {mode_label}.'
                ),
            )
        )

        if video_llm:
            message.append(
                dict(
                    type='text',
                    value=f'The accumulated context contains {len(context_videos)} chronological clip(s), merged into one video.',
                )
            )
            if merged_video_path:
                message.append(dict(type='video', value=merged_video_path))
            if self.use_audio and merged_audio_path:
                message.append(dict(type='audio', value=merged_audio_path))
        else:
            message.append(
                dict(
                    type='text',
                    value=f'The accumulated context contains {len(context_videos)} chronological clip(s), merged before frame sampling.',
                )
            )
            for im in self._save_clip_frames(merged_video_path):
                message.append(dict(type='image', value=im))
            if self.use_audio and merged_audio_path:
                message.append(dict(type='audio', value=merged_audio_path))

        history = self._history_text(line)
        if history:
            message.append(dict(type='text', value=self.HISTORY_TMPL.format(history=history)))

        choices = _load_json_str(line.get('candidates', '[]'), [])
        choices_str = '\n'.join([_safe_str(x) for x in choices])
        current_user_content = _load_json_str(line.get('current_user_content', '""'), '')
        raw_question, _ = _parse_user_content(current_user_content)
        question_prefix = ''
        if not raw_question:
            question_prefix = 'This step continues the current streaming question.\n'

        message.append(
            dict(
                type='text',
                value=self.QUESTION_TMPL.format(
                    turn_type=turn_type,
                    question_prefix=question_prefix,
                    question=_safe_str(line.get('question', '')).strip(),
                    choices=choices_str,
                ),
            )
        )
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        score_file = get_intermediate_file_path(eval_file, '_score')
        if not osp.exists(score_file):
            data = load(eval_file)
            if 'prediction' not in data.columns:
                raise KeyError('Missing `prediction` column in eval file.')

            scores = []
            pred_letters = []
            for _, row in data.iterrows():
                ans = _safe_str(row.get('answer', '')).strip().upper()
                pred = _safe_str(row.get('prediction', ''))
                pred_letter = _extract_choice_letter(pred)
                pred_letters.append(pred_letter)
                if not pred_letter:
                    scores.append(-1)
                else:
                    scores.append(int(pred_letter == ans))

            data = data.copy()
            data['pred_letter'] = pred_letters
            data['score'] = scores
            dump(data, score_file)

        data = load(score_file)
        total = int(len(data))
        valid_mask = data['score'] >= 0
        valid = int(valid_mask.sum())

        result = {
            'overall': _acc(list(map(int, data['score'].tolist()))),
            'total': total,
            'valid': valid,
        }

        def group_acc(col: str):
            if col not in data.columns:
                return {}
            out = {}
            for k, sub in data[valid_mask].groupby(col):
                out[_safe_str(k)] = float(np.mean(sub['score'].astype(float)))
            return out

        result['by_turn_type'] = group_acc('turn_type')
        result['by_group_size'] = group_acc('group_num_turns')
        result['by_session_length'] = group_acc('session_num_turns')

        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(result, rating_file)
        return result
