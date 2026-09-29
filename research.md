# Speech Research Papers 2025–2026

## Nhóm 1: Full-duplex / Realtime Speech Agent

| Time       | Bài báo                                                                          | Chủ đề                                                                                         |
| ---------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 14/09/2026 | **Enabling Streaming User Transcription in Full-Duplex Speech-to-Speech Models** | Thêm ASR realtime vào model full-duplex S2S, WER 10.21%.                                       |
| 08/09/2026 | **ConversationalVoice**                                                          | Tạo data full-duplex từ hội thoại thật, giữ overlap, interruption và backchannel.              |
| 06/07/2026 | **DuplexChat**                                                                   | Tạo corpus speaker-separated full-duplex cực lớn từ podcast.                                   |
| 12/06/2026 | **BayLing-Duplex**                                                               | SpeechLM full-duplex, một LLM tự quyết định nghe/nói/dừng, không cần module turn-taking riêng. |
| 23/04/2026 | **HumDial Challenge / Full-Duplex Interaction**                                  | Benchmark challenge về interrupt, overlap và turn-taking.                                      |
| 21/04/2026 | **UAF: Unified Audio Front-end LLM**                                             | Gom VAD, turn detection, speaker recognition, ASR và QA vào một audio front-end LLM.           |
| 06/2026    | **IRAF**                                                                         | Chống nhiễu/interference cho full-duplex spoken dialogue.                                      |

---

## Nhóm 2: ASR / Low-resource ASR / Multilingual ASR

| Time       | Bài báo                                                            | Chủ đề                                                                         |
| ---------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 14/09/2026 | **Sequential Adapter Stacking for Cross-Lingual Low-Resource ASR** | Stack adapter ngôn ngữ nguồn + target để giảm WER 5–8%.                        |
| 14/09/2026 | **Merging the Knowledge of LLMs for ASR**                          | Merge kiến thức LLM vào ASR bằng LoRA, không tăng cost inference.              |
| 03/09/2026 | **ChinaVoices Challenge 2026**                                     | ASR + nhận diện phương ngữ Trung Quốc, gồm 16 dialect categories.              |
| 02/09/2026 | **VibeVoice-ASR-Streaming Technical Report**                       | ASR streaming mới, submitted 02/09/2026, revised 10/09/2026.                   |
| 19/08/2026 | **A Speech Corpus for Mizo ASR**                                   | Fine-tune Whisper/SraVaani cho tiếng Mizo low-resource.                        |
| 16/03/2026 | **Vietnamese Automatic Speech Recognition: A Revisit**             | Đánh giá lại ASR tiếng Việt, nhấn mạnh vấn đề chất lượng dataset.              |
| 20/03/2026 | **LoASR-Bench**                                                    | Benchmark Speech-LLM cho ASR ngôn ngữ ít dữ liệu.                              |
| 25/02/2026 | **TG-ASR**                                                         | Dùng translation-guided learning + gated cross attention cho low-resource ASR. |
| 02/2026    | **Depth-Aware Adaptation for Multilingual Speech Models**          | Adapt model ASR đa ngôn ngữ hiệu quả hơn LoRA thường.                          |
| 11/01/2026 | **Task Arithmetic with Support Languages for Low-Resource ASR**    | Dùng task arithmetic từ ngôn ngữ hỗ trợ để cải thiện ASR low-resource.         |
| 09/2025    | **Frustratingly Easy Data Augmentation for Low-Resource ASR**      | Tạo text mới + TTS để augment data ASR ít tài nguyên.                          |

---

## Nhóm 3: Diarization / Speaker-Attributed ASR / Multi-Speaker ASR

| Time       | Bài báo                                                   | Chủ đề                                                                                      |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 09/09/2026 | **Pushing the Boundaries of Streaming Multi-Speaker ASR** | So sánh 4 kiến trúc streaming multi-speaker ASR về accuracy, latency và memory.             |
| 20/11/2025 | **Train Short, Infer Long / JEDIS-LLM**                   | Speech-LLM train audio ngắn nhưng infer streaming trên audio dài, hỗ trợ ASR + diarization. |
| 13/04/2026 | **Speaker Attributed ASR Using Speech Aware LLMs**        | ASR có gắn nhãn speaker trực tiếp trong transcript.                                         |
| 24/04/2026 | **DM-ASR**                                                | Diarization-aware multi-speaker ASR bằng LLM.                                               |
| 11/03/2026 | **G-STAR**                                                | Speaker-attributed ASR dài, giữ identity speaker xuyên chunk.                               |
| 01/2026    | **TagSpeech**                                             | End-to-end multi-speaker ASR + diarization bằng time anchor.                                |
| 18/09/2025 | **Identity-Aware LLM Refinement of Speaker Diarization**  | Dùng LLM refine diarization theo identity, không cần train.                                 |

---

## Nhóm 4: Target Speaker Extraction / Overlap Speech

| Time    | Bài báo                                                   | Chủ đề                                                             |
| ------- | --------------------------------------------------------- | ------------------------------------------------------------------ |
| 07/2026 | **SLT 2026 REAL-TSE Challenge**                           | Challenge TSE từ hội thoại thật.                                   |
| 07/2026 | **tttAI System for TSA-ASR SmartGlasses Challenge**       | Diarization + overlap detection + TSE + ASR cho smart glasses.     |
| 04/2026 | **Towards Streaming Target Speaker Extraction / StarTSE** | TSE streaming theo chunk, tập trung xử lý latency.                 |
| 03/2026 | **Training-Free Multi-Step Inference for TSE**            | TSE nhiều bước không cần train lại model.                          |
| 03/2026 | **Training Dynamics-Aware Curriculum for TSE**            | Curriculum learning theo dynamics trong quá trình train cho TSE.   |
| 05/2026 | **Flexible Multi-Channel Target Speaker Extraction**      | TSE nhiều microphone với spatial selective filter.                 |
| 03/2026 | **HRTF-guided Binaural TSE**                              | Dùng HRTF cho tách speaker trong audio binaural.                   |
| 03/2026 | **Plug-and-Steer**                                        | Audio-visual TSE, tách separation và selection.                    |
| 02/2026 | **Keyword Guided Target Speaker Extraction**              | Không cần enrollment sạch, dùng keyword để hướng dẫn tách speaker. |

---

## Nhóm 5: Speech Enhancement / Noise / Data Curation

| Time                     | Bài báo                                                            | Chủ đề                                                              |
| ------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 08/2026                  | **Training DeepFilterNet with Accurate Room Acoustic Simulations** | Mô phỏng phòng chính xác hơn giúp train speech enhancement tốt hơn. |
| 26/01/2026               | **Geneses: Unified Generative Speech Enhancement and Separation**  | Một model generative cho cả enhancement và separation.              |
| 23/01/2026               | **FlowSE-GRPO**                                                    | Train speech enhancement bằng online reinforcement learning.        |
| 18/01/2026               | **Confidence-based Filtering for Speech Dataset Curation**         | Lọc dataset speech bằng confidence + generative enhancement.        |
| 09/2025                  | **One-Step Generative Speech Enhancement via MeanFlow**            | Speech enhancement một bước, hướng generative nhanh.                |
| 07/2025, revised 05/2026 | **Robust One-step Speech Enhancement via Consistency**             | Enhancement một bước bằng consistency model.                        |

---

## Nhóm 6: TTS / Speech Quality / Speech Assessment

| Time       | Bài báo                                       | Chủ đề                                                              |
| ---------- | --------------------------------------------- | ------------------------------------------------------------------- |
| 14/09/2026 | **OpenEnded**                                 | Corpus chấm speaking proficiency theo accuracy, fluency và prosody. |
| 12/09/2026 | **VoiceMOS Challenge 2026**                   | Đánh giá speech enhancement, emotional TTS và accented TTS.         |
| 10/09/2026 | **Nuha-Speech**                               | Xây Speech-LLM tiếng Ả Rập với 1.5M speech QA samples.              |
| 08/2026    | **EmoTra-TTS**                                | TTS có khả năng chuyển cảm xúc mượt trong cùng utterance.           |
| 05/02/2026 | **Zero-Shot TTS With Enhanced Audio Prompts** | TTS zero-shot dùng audio prompt đã được enhancement.                |
| 09/2025    | **Sidon**                                     | Open-source multilingual speech generation/TTS nhanh và robust.     |
