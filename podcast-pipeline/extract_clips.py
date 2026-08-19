import json
import os
import argparse
from pydub import AudioSegment

def load_segments(json_path: str):
    """Đọc danh sách segment từ file JSON của pipeline.

    Pipeline xuất {"metadata": ..., "segments": [...]}, còn các bản dump trung
    gian là list thuần, nên chấp nhận cả hai dạng.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        segments = data.get("segments")
        if segments is None:
            raise ValueError(
                f"{json_path} là JSON object nhưng không có khoá 'segments'. "
                f"Các khoá hiện có: {sorted(data)}"
            )
        return segments
    if isinstance(data, list):
        return data
    raise ValueError(f"{json_path} phải là list segment hoặc object có 'segments'.")


def extract_dialogue_clips(json_path: str, audio_path: str, output_dir: str, min_turns: int = 5,
                           max_silence: float = 3.0, pan_stereo: bool = False):
    """
    Tự động quét file JSON tìm các đoạn hội thoại "ping-pong" (luân phiên 2 người nói)
    và dùng pydub để cắt file Audio gốc thành các clip ngắn.
    """
    print(f"Loading JSON: {json_path}")
    segments = load_segments(json_path)

    if not segments:
        print("JSON trống!")
        return

    # Thuật toán Gom nhóm (Sliding Window)
    clips = []
    current_clip = [segments[0]]

    for i in range(1, len(segments)):
        prev_seg = segments[i-1]
        curr_seg = segments[i]

        # Kiểm tra khoảng lặng giữa 2 câu
        silence_gap = curr_seg['start'] - prev_seg['end']

        # Nếu đổi người nói VÀ khoảng lặng không quá lớn -> Cùng 1 đoạn hội thoại
        if curr_seg['speaker'] != prev_seg['speaker'] and silence_gap <= max_silence:
            current_clip.append(curr_seg)
        else:
            # Nếu người nói không đổi (nói 1 lèo quá dài) hoặc khoảng lặng quá lớn -> Cắt Block
            if len(current_clip) >= min_turns:
                clips.append(current_clip)

            # Khởi tạo Block mới. Nếu chỉ vì cùng người nói (không phải do lặng
            # quá lâu) thì prev_seg vẫn thuộc mạch hội thoại mới, giữ lại để clip
            # không bị cụt đầu.
            same_speaker_break = curr_seg['speaker'] == prev_seg['speaker']
            if same_speaker_break and silence_gap <= max_silence:
                current_clip = [prev_seg, curr_seg]
            else:
                current_clip = [curr_seg]

    # Chốt Block cuối cùng nếu vòng lặp kết thúc
    if len(current_clip) >= min_turns:
        clips.append(current_clip)

    print(f"Tìm thấy {len(clips)} phân đoạn hội thoại đạt chuẩn (>= {min_turns} lượt lời)!")
    
    if not clips:
        return
        
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Audio: {audio_path} (Quá trình này mất vài giây...)")
    try:
        audio = AudioSegment.from_file(audio_path)
        # Ép về âm thanh Stereo (2 kênh) để có thể Pan trái/phải
        if audio.channels == 1:
            audio = AudioSegment.from_mono_audiosegments(audio, audio)
    except Exception as e:
        print(f"Lỗi đọc file Audio: {e}")
        return

    for idx, clip in enumerate(clips):
        start_time_sec = clip[0]['start']
        end_time_sec = clip[-1]['end']
        
        start_ms = max(0, int((start_time_sec - 0.5) * 1000))
        end_ms = min(len(audio), int((end_time_sec + 0.5) * 1000))
        
        print(f"Đang cắt Clip {idx+1}: Từ {start_time_sec:.1f}s đến {end_time_sec:.1f}s ...")
        
        if pan_stereo:
            # Tạo một track im lặng làm nền
            clip_duration_ms = end_ms - start_ms
            final_mix = AudioSegment.silent(duration=clip_duration_ms, frame_rate=audio.frame_rate)
            # Ép về Stereo
            final_mix = AudioSegment.from_mono_audiosegments(final_mix, final_mix) if final_mix.channels == 1 else final_mix
            
            for s in clip:
                seg_start_ms = int(s['start'] * 1000)
                seg_end_ms = int(s['end'] * 1000)
                
                # Cắt đoạn giọng nói của người này
                voice_chunk = audio[seg_start_ms:seg_end_ms]
                
                # Pan âm thanh: Lẻ (VD: SPEAKER_00) sang Trái 80%, Chẵn (SPEAKER_01) sang Phải 80%
                if "00" in s['speaker'] or "02" in s['speaker']:
                    voice_chunk = voice_chunk.pan(-0.8) # -1.0 là trái 100%
                else:
                    voice_chunk = voice_chunk.pan(0.8)  # 1.0 là phải 100%
                    
                # Overlay lên track mix tại đúng vị trí timestamp tương đối
                rel_pos = seg_start_ms - start_ms
                final_mix = final_mix.overlay(voice_chunk, position=rel_pos)
                
            extracted_audio = final_mix
        else:
            extracted_audio = audio[start_ms:end_ms]
        
        out_file = os.path.join(output_dir, f"viral_clip_{idx+1}.mp3")
        extracted_audio.export(out_file, format="mp3", bitrate="192k")
        
        # Xuất luôn file Transcript phụ đề đi kèm cho Clip đó
        out_txt = os.path.join(output_dir, f"viral_clip_{idx+1}.txt")
        with open(out_txt, 'w', encoding='utf-8') as tf:
            for s in clip:
                tf.write(f"[{s['speaker']}] {s.get('text', '')}\n")
                
    print(f"Hoàn tất! Các file đã được lưu tại thư mục: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Đường dẫn file JSON đầu ra của Pipeline")
    parser.add_argument("--audio", required=True, help="Đường dẫn file MP3/WAV gốc")
    parser.add_argument("--out", default="./viral_clips", help="Thư mục chứa các clip xuất ra")
    parser.add_argument("--turns", type=int, default=5, help="Số lượt lời qua lại tối thiểu để tạo thành 1 clip")
    parser.add_argument("--pan_stereo", action="store_true", help="Bật hiệu ứng âm thanh ASMR: Speaker 1 tai trái, Speaker 2 tai phải")
    parser.add_argument("--max_silence", type=float, default=3.0, help="Khoảng lặng tối đa (giây) giữa 2 lượt lời trong cùng một clip")
    args = parser.parse_args()

    extract_dialogue_clips(
        args.json, args.audio, args.out,
        min_turns=args.turns,
        max_silence=args.max_silence,
        pan_stereo=args.pan_stereo,
    )
