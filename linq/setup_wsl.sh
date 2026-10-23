#!/usr/bin/env bash
# Установка SDK LinQ в WSL и компиляция нейросетей ДОЗОР под LinQ TPU.
# Запускается из linq_setup.bat (двойной клик). Можно перезапускать: готовые шаги пропускаются.
set -o pipefail
HACK="$(pwd)"
LOG="$HACK/linq_setup.log"
exec > >(tee -a "$LOG") 2>&1
W=/root/linqwork
V=/root/linq
step() { echo; echo "=================== $* ($(date +%H:%M:%S))"; }
fail() { echo; echo "!!!!!! ОШИБКА: $*"; echo "Пришлите файл linq_setup.log в чат"; echo "STATUS: FAILED"; exit 1; }

step "0. Где мы"
echo "Папка: $HACK"; uname -a; . /etc/os-release; echo "$PRETTY_NAME"; df -h / | tail -1; free -g | head -2
[ -d "$HACK/tpu_framework/packets" ] || fail "нет папки tpu_framework/packets рядом со скриптом"
[ -d "$HACK/safety_vision" ] || fail "нет папки safety_vision рядом со скриптом"

step "1. Системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q || fail "apt update (нет интернета? выключите VPN хакатона на время установки)"
if ! apt-cache show python3.12-venv >/dev/null 2>&1; then
  apt-get install -y -q software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -q
fi
apt-get install -y -q python3.12 python3.12-venv libglib2.0-0 libgl1 || fail "apt install python3.12"
python3.12 --version

step "2. Окружение Python"
[ -x $V/bin/python ] || python3.12 -m venv $V || fail "venv"
. $V/bin/activate
pip install -q -U pip

step "3. Копирую файлы в Linux"
mkdir -p $W
cp -ru "$HACK/tpu_framework/packets" $W/ || fail "копирование packets"
rm -rf $W/safety_vision && cp -r "$HACK/safety_vision" $W/ || fail "копирование safety_vision"

step "4. PyTorch (CPU-версия)"
python -c "import torch" 2>/dev/null || pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu || fail "torch"

step "5. SDK LinQ (долго: TensorFlow и др., 20-40 мин)"
cd /root
if ! python -c "from tpu_framework import RegularModel" 2>/dev/null; then
  cd $W/packets && pip install -r requirements.txt || fail "установка SDK (requirements.txt)"
fi
pip install -q "numpy==1.26.4" "opencv-python-headless==4.10.0.84" pyyaml || fail "opencv"  # SDK требует numpy 1.26
cd /root && python -c "from tpu_framework import RegularModel, compiler; import onnx; print('SDK OK')" || fail "SDK не импортируется"

step "6. Компиляция нейросетей под LinQ"
cd $W/safety_vision
for m in yolo11n-pose yolo11n; do
  ls models/$m.head_b1_*_h1m.tpu >/dev/null 2>&1 && { echo "$m уже скомпилирована"; continue; }
  echo "--- $m"
  python linq/compile_linq.py models/$m.head.onnx --calib data --n ${CALIB_N:-48} || fail "компиляция $m"
done

step "7. Копирую результат обратно в Windows"
cp -v models/*.tpu "$HACK/safety_vision/models/" || fail "копирование .tpu"
echo; echo "ГОТОВО. Файлы .tpu лежат в safety_vision/models/"; echo "STATUS: DONE"
