# 공식 REPA 코드와 `src/train_align.py` 비교

대조한 파일: 공식 저장소(github.com/sihyun-yu/REPA)의 `train.py`, `loss.py`, `models/sit.py`, `utils.py`.

## 같은 부분

| 항목 | 공식 REPA | train_align.py |
|---|---|---|
| 정렬 위치 | `encoder_depth=8` (8번째 블록 출력) | `blocks[7]` 출력 (hook) |
| projector | Linear(h,2048)-SiLU-Linear(2048,2048)-SiLU-Linear(2048,z_dim) | 동일 |
| 정렬 loss | 두 벡터를 L2 정규화 후 내적의 음수 (= −cosine similarity) | 동일 |
| λ (`proj_coeff`) | 0.5 | 0.5 |
| 최종 loss | denoising + 0.5 × proj | l_simple + 0.5 × l_align |
| 옵티마이저 | AdamW, lr 1e-4, weight decay 0, betas (0.9, 0.999) | AdamW, lr 1e-4, wd 0 (betas 는 PyTorch 기본값과 동일) |
| DINOv2 입력 | ImageNet 정규화 후 224px bicubic 리사이즈 | 동일 |
| DINOv2 출력 | `x_norm_patchtokens` (패치 토큰) | `last_hidden_state[:,1:]` (CLS 제외 패치 토큰) |

## 다른 부분 (한계)

| 항목 | 공식 REPA | train_align.py |
|---|---|---|
| DINOv2 크기 | 기본 `dinov2-vit-b` (`torch.hub`) | `dinov2-small` (HuggingFace `Dinov2Model`) — T4 자원 때문. 두 출력의 수치 일치는 확인하지 않음 |
| 보간 경로 방향 | t=0 데이터, t=1 노이즈 | SiT transport: t=0 노이즈, t=1 데이터 (선형 경로로 동등, 표기만 반대) |
| CFG 학습용 label dropout | `cfg_prob=0.1` | SiT 기본 설정 (값 미확인) |
| VAE latent | 사전 추출 latent 사용 | 매 스텝 이미지에서 VAE 인코딩 |
| EMA / 체크포인트 / 주기적 샘플링 | 있음 | 없음 |
| 평가 지표 | FID 등 | 고정 노이즈 L_simple (학습셋·held-out) |
| 규모 | batch 256, ImageNet 전체 | batch 4, 40장, 300스텝 |

따라서 이 구현은 **"공식 REPA 설정을 따른 소규모 재현(REPA-style)"** 으로 표기한다.
