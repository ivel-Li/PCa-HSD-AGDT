#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# download_datasets.sh — 下载 7 个公开前列腺 MRI 数据集
# ═══════════════════════════════════════════════════════════════════════
#
# 数据集列表:
#   1. PROMISE12     — MICCAI 2012 前列腺分割挑战
#   2. NCI-ISBI 2013 — 前列腺结构自动分割挑战
#   3. I2CVB         — 多参数前列腺 MRI（部分需申请）
#   4. PROSTATEx     — SPIE-AAPM-NCI 前列腺 MR 分类挑战
#   5. MSD Prostate  — Medical Segmentation Decathlon Task05
#   6. Prostate158   — 双参数 3T 前列腺 MRI（带专家标注）
#   7. PI-CAI        — Prostate Imaging: Cancer AI 挑战
#
# 用法:
#   chmod +x download_datasets.sh
#   ./download_datasets.sh                    # 下载所有数据集
#   ./download_datasets.sh --dataset promise12  # 只下载指定数据集
#   ./download_datasets.sh --skip-existing     # 跳过已存在的文件
#
# 注意: 部分数据集（I2CVB, PI-CAI）需要手动申请权限。
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOAD_DIR="${BASE_DIR}/dataset/downloads"
mkdir -p "${DOWNLOAD_DIR}"

# 日志颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 已下载标记
SKIP_EXISTING=false
REQUESTED_DATASET=""

# ── 参数解析 ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)       REQUESTED_DATASET="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=true; shift ;;
    --help)          sed -n '/^#/p; /^$/q' "$0" | head -n -1; exit 0 ;;
    *)               warn "未知参数: $1"; shift ;;
  esac
done

# ── 工具函数 ─────────────────────────────────────────────────────────
should_download() {
  [[ -z "${REQUESTED_DATASET}" ]] && return 0
  [[ "${REQUESTED_DATASET}" == "$1" ]] && return 0
  return 1
}

download_file() {
  local url="$1" out_path="$2" desc="$3"
  if [[ -f "${out_path}" && "${SKIP_EXISTING}" == true ]]; then
    info "跳过已存在: ${out_path##*/}"
    return 0
  fi
  info "下载 ${desc} → ${out_path##*/}"
  mkdir -p "$(dirname "${out_path}")"
  if command -v wget &>/dev/null; then
    wget -c --show-progress -O "${out_path}" "${url}"
  elif command -v curl &>/dev/null; then
    curl -fL -o "${out_path}" "${url}"
  else
    error "需要 wget 或 curl"
    return 1
  fi
}

# ═══════════════════════════════════════════════════════════════════════
# 1. PROMISE12 — Zenodo
# ═══════════════════════════════════════════════════════════════════════
download_promise12() {
  local dir="${DOWNLOAD_DIR}/PROMISE12"
  mkdir -p "${dir}"

  # Zenodo record (updated version 2023-06-07)
  local base_url="https://zenodo.org/records/8026660/files"

  download_file "${base_url}/training_data.zip"   "${dir}/training_data.zip"   "PROMISE12 训练集"
  download_file "${base_url}/test_data.zip"        "${dir}/test_data.zip"       "PROMISE12 测试集"
  download_file "${base_url}/livechallenge_test_data.zip" "${dir}/livechallenge_test_data.zip" "PROMISE12 在线挑战测试集"
  download_file "${base_url}/LICENSE.TXT"          "${dir}/LICENSE.TXT"         "PROMISE12 许可证"

  info "PROMISE12 下载完成 → ${dir}"
  info "  请解压: cd ${dir} && unzip training_data.zip && unzip test_data.zip"
}

# ═══════════════════════════════════════════════════════════════════════
# 2. NCI-ISBI 2013 — TCIA (via NBIA)
# ═══════════════════════════════════════════════════════════════════════
download_nci_isbi_2013() {
  local dir="${DOWNLOAD_DIR}/NCI-ISBI2013"
  mkdir -p "${dir}"

  info "NCI-ISBI 2013 数据可从 TCIA 获取:"
  info "  https://wiki.cancerimagingarchive.net/display/public/nci-isbi+2013+challenge"
  info ""
  info "推荐使用 NBIA Data Retriever 下载:"
  info "  1. 访问: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images"
  info "  2. 下载并安装 NBIA Data Retriever"
  info "  3. 下载清单文件 (manifest):"
  info "     https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013"
  info "  4. 使用 NBIA Data Retriever 打开 .tcia 文件下载"
  info ""
  info "或者使用 TCIA 的 REST API 下载（需要先获取集合 ID）:"
  info "  数据集名称: PROSTATE-DIAGNOSIS"
  info "  下载命令示例:"
  info '    pip install tciaclient'
  info '    python -c "from tciaclient import TCIA; t = TCIA(); t.downloadSeries(\"PROSTATE-DIAGNOSIS\", outputDir=\"'"${dir}"'\")"'
  info ""
  warn "NCI-ISBI 2013 暂未实现自动下载（TCIA 需交互认证），请手动下载。"
}

# ═══════════════════════════════════════════════════════════════════════
# 3. I2CVB — Zenodo (受限) / GitHub 处理代码
# ═══════════════════════════════════════════════════════════════════════
download_i2cvb() {
  local dir="${DOWNLOAD_DIR}/I2CVB"
  mkdir -p "${dir}"

  info "I2CVB 多参数前列腺 MRI 数据:"
  info "  Zenodo 记录: https://zenodo.org/records/162228"
  info "  (需要申请访问权限，非公开直接下载)"
  info ""
  info "  官方网页: https://i2cvb.github.io/"
  info "  处理代码: https://github.com/I2Cvb/mp-mri-prostate"
  info ""
  info "数据说明:"
  info "  - 包含 T2-W MRI, DCE-MRI, DWI, ADC, MRSI"
  info "  - 来自 Siemens 3T 和 Siemens 1.5T 两种扫描仪"
  info "  - 格式: DICOM"
  info "  - 含金标准标注 (DICOM 格式)"
  info ""
  info "部分数据子集已包含于其他公开来源:"
  info "  1. T2-W-MRI, DCE-MRI, ADC, MRSI (Siemens 3T): 参见文献 [2]"
  info "  2. T2-W-MRI, DCE-MRI (Siemens 3T): 参见文献 [3]"
  info ""
  warn "I2CVB 需要向作者申请数据访问权限，请访问官网提交申请。"
  warn "或将申请邮件发送至 i2cvb@lpcv.ua.es"
}

# ═══════════════════════════════════════════════════════════════════════
# 4. PROSTATEx — TCIA (已并入 PI-CAI)
# ═══════════════════════════════════════════════════════════════════════
download_prostatex() {
  local dir="${DOWNLOAD_DIR}/PROSTATEx"
  mkdir -p "${dir}"

  info "PROSTATEx 数据集:"
  info "  TCIA 集合: https://www.cancerimagingarchive.net/collection/prostatex"
  info ""
  info "注意: PROSTATEx (训练集 + 测试集) 已包含在 PI-CAI 公开训练数据集中。"
  info "建议直接下载 PI-CAI 数据集 (见下) 以同时获得 PROSTATEx 数据。"
  info ""
  info "如需单独下载 PROSTATEx:"
  info "  1. 访问 TCIA 页面: https://www.cancerimagingarchive.net/collection/prostatex"
  info "  2. 点击 'Download' 获取 NBIA Data Retriever manifest 文件"
  info "  3. 使用 NBIA Data Retriever 下载"
  info ""
  info "PROSTATEx 掩码 (第三方): https://rcuocolo.github.io/PROSTATEx_masks/"
  info ""
  warn "建议跳过 PROSTATEx 单独下载，直接下载 PI-CAI 数据集。"
}

# ═══════════════════════════════════════════════════════════════════════
# 5. MSD Prostate (Task05) — Google Drive / AWS
# ═══════════════════════════════════════════════════════════════════════
download_msd_prostate() {
  local dir="${DOWNLOAD_DIR}/MSD_Prostate/Task05_Prostate"
  mkdir -p "${dir}"

  info "MSD Prostate (Task05) 下载方式:"
  info ""
  info "方式 1: Google Drive (官方推荐)"
  info "  链接: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2"
  info "  文件名: Task05_Prostate.tar"
  info "  大小: ~0.8 GB"
  info ""
  info "方式 2: AWS S3 (更快)"
  info "  命令:"
  info "    pip install awscli"
  info "    aws s3 sync --no-sign-request s3://msd-for-monai/Task05_Prostate/ ${dir}/"
  info ""
  info "方式 3: 使用 MSD 官方脚本来 python 下载:"
  info '    python -c "
import requests, tarfile, os
url = \"https://msd-for-monai.s3.us-west-2.amazonaws.com/Task05_Prostate.tar\"'
  info '    r = requests.get(url, stream=True)'

  # 尝试通过 AWS S3 (无签名) 直接下载
  info "正在通过 AWS S3 (无认证) 下载 MSD Prostate ..."
  if command -v aws &>/dev/null; then
    aws s3 sync --no-sign-request s3://msd-for-monai/Task05_Prostate/ "${dir}/" 2>/dev/null || {
      warn "AWS CLI 下载失败，请尝试其他方式。"
    }
  else
    # 使用 curl 下载 tar 文件
    local tar_path="${DOWNLOAD_DIR}/MSD_Prostate/Task05_Prostate.tar"
    download_file \
      "https://msd-for-monai.s3.us-west-2.amazonaws.com/Task05_Prostate.tar" \
      "${tar_path}" "MSD Prostate (Task05)"
    info "下载完成! 请解压: cd ${DOWNLOAD_DIR}/MSD_Prostate && tar xf Task05_Prostate.tar"
  fi
}

# ═══════════════════════════════════════════════════════════════════════
# 6. Prostate158 — Zenodo
# ═══════════════════════════════════════════════════════════════════════
download_prostate158() {
  local dir="${DOWNLOAD_DIR}/Prostate158"
  mkdir -p "${dir}"

  info "下载 Prostate158 训练集 (139 MRI) ..."
  local train_doi="10.5281/zenodo.6481141"
  # Zenodo 直接下载链接 (通过 DOI 解析)
  local train_url="https://zenodo.org/records/6481141/files/Prostate158_train.zip"
  local test_url="https://zenodo.org/records/6592345/files/Prostate158_test.zip"

  download_file "${train_url}" "${dir}/Prostate158_train.zip" "Prostate158 训练集"
  download_file "${test_url}"  "${dir}/Prostate158_test.zip"  "Prostate158 测试集 (19 MRI)"

  info "Prostate158 下载完成 → ${dir}"
  info "  请解压: cd ${dir} && unzip Prostate158_train.zip && unzip Prostate158_test.zip"
}

# ═══════════════════════════════════════════════════════════════════════
# 7. PI-CAI — Zenodo / Grand Challenge
# ═══════════════════════════════════════════════════════════════════════
download_picai() {
  local dir="${DOWNLOAD_DIR}/PI-CAI"
  mkdir -p "${dir}"

  info "PI-CAI 公开训练数据集 (1500 例):"
  info "  Zenodo DOI: 10.5281/zenodo.6517397"
  info "  https://zenodo.org/records/6517397"
  info ""
  info "  下载方式:"
  info "    pip install picai"
  info "    python -c \"from picai_dataset import download; download.picai_public()\""
  info ""
  info " 或直接使用 Zenodo 下载:"
  info "    wget https://zenodo.org/records/6517397/files/picai_public_train.zip"
  info "    wget https://zenodo.org/records/6517397/files/picai_public_validation.zip"
  info ""
  info "更多信息: https://pi-cai.grand-challenge.org/DATA/"
  info ""
  info "PI-CAI 数据包含:"
  info "  - T2W, DWI, ADC 序列"
  info "  - csPCa 标注 (活检确认)"
  info "  - AI 衍生的病变分割"
  info "  - 包含所有 PROSTATEx 病例"
  info ""

  # 尝试使用 picai 包下载
  if python -c "import picai_dataset" 2>/dev/null; then
    info "使用 picai 库下载 PI-CAI 数据集..."
    python -c "
from picai_dataset import download
download.picai_public(output_dir='${dir}')
" 2>&1 || warn "picai 下载失败，请手动下载。"
  else
    warn "未安装 picai 库。尝试通过 Zenodo 直接下载..."
    warn "请先安装: pip install picai"
    warn "或从 Zenodo 手动下载: https://zenodo.org/records/6517397"
  fi

  cat << 'EOF' > "${dir}/README.md"
# PI-CAI 数据集

## 数据访问

PI-CAI 数据集的访问分为几个部分:

### 公开训练数据 (1500 例)
- 任何人可下载: https://pi-cai.grand-challenge.org/DATA/
- Zenodo DOI: 10.5281/zenodo.6517397

### 私有/隔离训练数据 (7607 例)
- 仅限挑战组织者使用
- 用于重新训练前 5 名的 AI 算法

### 隐藏调优队列 (100 例)
- 用于公开排行榜

### 隐藏测试队列 (1000 例)
- 用于最终评估

## 引用
```
@article{sauter2024artificial,
  title={Artificial Intelligence and Radiologists in Prostate Cancer Detection on MRI (PI-CAI): An International, Paired, Non-Inferiority, Confirmatory Study},
  author={Sauter, Anindo and others},
  journal={The Lancet Oncology},
  year={2024}
}
```
EOF
  info "PI-CAI README 已写入 ${dir}/README.md"
}

# ═══════════════════════════════════════════════════════════════════════
# 主执行流程
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       前列腺 MRI 公开数据集下载脚本                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 检查和安装依赖
NEEDED_CMDS=()
for cmd in wget curl; do
  command -v "$cmd" &>/dev/null || NEEDED_CMDS+=("$cmd")
done
if [[ ${#NEEDED_CMDS[@]} -gt 0 ]]; then
  error "缺少命令: ${NEEDED_CMDS[*]}"
  info "请安装: apt install ${NEEDED_CMDS[*]}"
  exit 1
fi

# ── PROMISE12 ─────────────────────────────────────────────────────────
if should_download "promise12"; then
  echo "───────────────────────────────────────────────────────────────"
  info "1/7  PROMISE12"
  echo "───────────────────────────────────────────────────────────────"
  download_promise12
  echo ""
fi

# ── NCI-ISBI 2013 ────────────────────────────────────────────────────
if should_download "nci-isbi2013"; then
  echo "───────────────────────────────────────────────────────────────"
  info "2/7  NCI-ISBI 2013"
  echo "───────────────────────────────────────────────────────────────"
  download_nci_isbi_2013
  echo ""
fi

# ── I2CVB ─────────────────────────────────────────────────────────────
if should_download "i2cvb"; then
  echo "───────────────────────────────────────────────────────────────"
  info "3/7  I2CVB"
  echo "───────────────────────────────────────────────────────────────"
  download_i2cvb
  echo ""
fi

# ── PROSTATEx ─────────────────────────────────────────────────────────
if should_download "prostatex"; then
  echo "───────────────────────────────────────────────────────────────"
  info "4/7  PROSTATEx"
  echo "───────────────────────────────────────────────────────────────"
  download_prostatex
  echo ""
fi

# ── MSD Prostate ──────────────────────────────────────────────────────
if should_download "msd-prostate"; then
  echo "───────────────────────────────────────────────────────────────"
  info "5/7  MSD Prostate (Task05)"
  echo "───────────────────────────────────────────────────────────────"
  download_msd_prostate
  echo ""
fi

# ── Prostate158 ───────────────────────────────────────────────────────
if should_download "prostate158"; then
  echo "───────────────────────────────────────────────────────────────"
  info "6/7  Prostate158"
  echo "───────────────────────────────────────────────────────────────"
  download_prostate158
  echo ""
fi

# ── PI-CAI ────────────────────────────────────────────────────────────
if should_download "picai"; then
  echo "───────────────────────────────────────────────────────────────"
  info "7/7  PI-CAI"
  echo "───────────────────────────────────────────────────────────────"
  download_picai
  echo ""
fi

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  下载脚本执行完毕                                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "下载目录: ${DOWNLOAD_DIR}"
echo ""
echo "已实现自动下载的数据集:"
echo "  ✅ PROMISE12     — 通过 Zenodo 直接下载"
echo "  ✅ MSD Prostate  — 通过 AWS S3 直接下载"
echo "  ✅ Prostate158   — 通过 Zenodo 直接下载"
echo "  ✅ PI-CAI        — 通过 picai 库尝试下载"
echo ""
echo "需要手动获取的数据集:"
echo "  ⚠️  NCI-ISBI 2013 — 需通过 TCIA NBIA Data Retriever"
echo "  ⚠️  I2CVB         — 需申请访问权限 (i2cvb@lpcv.ua.es)"
echo "  ⚠️  PROSTATEx     — 已包含于 PI-CAI 数据集，建议直接使用 PI-CAI"
echo ""
echo "解压说明:"
echo "  cd ${DOWNLOAD_DIR}/PROMISE12 && for f in *.zip; do unzip \"\$f\"; done"
echo "  cd ${DOWNLOAD_DIR}/MSD_Prostate && tar xf Task05_Prostate.tar"
echo "  cd ${DOWNLOAD_DIR}/Prostate158 && for f in *.zip; do unzip \"\$f\"; done"
echo ""
echo "更多信息请参考项目 README.md"