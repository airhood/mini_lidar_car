# Mini LiDAR Car

2D LiDAR를 이용한 소형 실내 차동 구동 로봇의 장애물 회피 연구. Unity에서 주행 데이터를 수집하고 PyTorch로 제어 정책을 학습했다.

- [연구 기록 및 평가](docs/policy-evaluation.md)
- [데이터 생성](docs/data-generation.md)
- [소프트웨어 연구 요약](docs/steam-report-software.md)

- [성과발표회 보고서 PDF](output/pdf/STEAM%20성과발표회%20결과보고서%20미리보기.pdf)
- [완주 추적 영상](experiments/doorway-continuous-switch/follow-video/success-follow-camera.mp4)
- [조향과 LiDAR 노이즈 영상](experiments/doorway-continuous-switch/steering-noise-video/success-steering-noise.mp4)
- [실험 결과와 로그](experiments)

Unity 프로젝트는 `mini_lidar_car-sim` 폴더에 있다.

[원본 LiDAR CSV 및 렌더링 프레임](https://github.com/airhood/mini_lidar_car/releases/tag/research-data-v1)은 저장소 루트에서 `tar --zstd -xf 파일명.tar.zst`로 풀면 원래 경로로 복원된다. Release의 `SHA256SUMS.txt`에 압축 파일의 체크섬을 기록했다.
