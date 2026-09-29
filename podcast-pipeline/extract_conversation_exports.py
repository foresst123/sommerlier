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


def extract_conversation_exports(json_path: str, audio_path: str, output_dir: str, min_turns: int = 5,
                           max_silence: float = 3.0, pan_stereo: bool = False):
    """
    Tự động quét file JSON tìm các đoạn hội thoại "ping-pong" (luân phiên 2 người nói)
    và dùng pydub để xuất các đoạn audio ngắn.
    """
    print(f"Loading JSON: {json_path}")
    segments = load_segments(json_path)

    if not segments:
        print("JSON trống!")
        return

    # Thuật toán Gom nhóm (Sliding Window)
    exports = []
    current_export = [segments[0]]

    for i in range(1, len(segments)):
        prev_seg = segments[i-1]
        curr_seg = segments[i]

        # Kiểm tra khoảng lặng giữa 2 câu
        silence_gap = curr_seg['start'] - prev_seg['end']

        # Nếu đổi người nói VÀ khoảng lặng không quá lớn -> Cùng 1 đoạn hội thoại
        if curr_seg['speaker'] != prev_seg['speaker'] and silence_gap <= max_silence:
            current_export.append(curr_seg)
        else:
            # Nếu người nói không đổi (nói 1 lèo quá dài) hoặc khoảng lặng quá lớn -> Cắt Block
            if len(current_export) >= min_turns:
                exports.append(current_export)

            # Khởi tạo Block mới. Nếu chỉ vì cùng người nói (không phải do lặng
            # quá lâu) thì prev_seg vẫn thuộc mạch hội thoại mới, giữ lại để đoạn xuất
            # không bị cụt đầu.
            same_speaker_break = curr_seg['speaker'] == prev_seg['speaker']
            if same_speaker_break and silence_gap <= max_silence:
                current_export = [prev_seg, curr_seg]
            else:
                current_export = [curr_seg]

    # Chốt Block cuối cùng nếu vòng lặp kết thúc
    if len(current_export) >= min_turns:
        exports.append(current_export)

    print(f"Tìm thấy {len(exports)} phân đoạn hội thoại đạt chuẩn (>= {min_turns} lượt lời)!")
    
    if not exports:
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

    for idx, excerpt in enumerate(exports):
        start_time_sec = excerpt[0]['start']
        end_time_sec = excerpt[-1]['end']
        
        start_ms = max(0, int((start_time_sec - 0.5) * 1000))
        end_ms = min(len(audio), int((end_time_sec + 0.5) * 1000))
        
        print(f"Đang xuất đoạn {idx+1}: Từ {start_time_sec:.1f}s đến {end_time_sec:.1f}s ...")
        
        if pan_stereo:
            # Tạo một track im lặng làm nền
            excerpt_duration_ms = end_ms - start_ms
            final_mix = AudioSegment.silent(duration=excerpt_duration_ms, frame_rate=audio.frame_rate)
            # Ép về Stereo
            final_mix = AudioSegment.from_mono_audiosegments(final_mix, final_mix) if final_mix.channels == 1 else final_mix
            
            for s in excerpt:
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
        
        out_file = os.path.join(output_dir, f"conversation_export_{idx+1}.mp3")
        extracted_audio.export(out_file, format="mp3", bitrate="192k")
        
        # Xuất transcript đi kèm cho đoạn này.
        out_txt = os.path.join(output_dir, f"conversation_export_{idx+1}.txt")
        with open(out_txt, 'w', encoding='utf-8') as tf:
            for s in excerpt:
                tf.write(f"[{s['speaker']}] {s.get('text', '')}\n")
                
    print(f"Hoàn tất! Các file đã được lưu tại thư mục: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Đường dẫn file JSON đầu ra của Pipeline")
    parser.add_argument("--audio", required=True, help="Đường dẫn file MP3/WAV gốc")
    parser.add_argument("--out", default="./conversation_exports", help="Thư mục chứa các đoạn xuất ra")
    parser.add_argument("--turns", type=int, default=5, help="Số lượt lời qua lại tối thiểu để tạo thành một đoạn xuất")
    parser.add_argument("--pan_stereo", action="store_true", help="Bật hiệu ứng âm thanh ASMR: Speaker 1 tai trái, Speaker 2 tai phải")
    parser.add_argument("--max_silence", type=float, default=3.0, help="Khoảng lặng tối đa giữa hai lượt lời trong cùng đoạn xuất")
    args = parser.parse_args()

    extract_conversation_exports(
        args.json, args.audio, args.out,
        min_turns=args.turns,
        max_silence=args.max_silence,
        pan_stereo=args.pan_stereo,
    )
