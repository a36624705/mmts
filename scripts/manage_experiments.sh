#!/usr/bin/env bash
# 实验管理工具（bash版本）
# 提供实验列表、清理、迁移等功能

set -euo pipefail

EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"
OLD_OUTPUTS_DIR="${OLD_OUTPUTS_DIR:-outputs}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 简单的JSON值提取函数
extract_json_value() {
    local json_file="$1"
    local key="$2"
    local default="$3"
    
    if [[ ! -f "$json_file" ]]; then
        echo "$default"
        return
    fi
    
    # 使用grep和sed提取JSON值
    local value=$(grep -o "\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$json_file" 2>/dev/null | sed 's/.*:[[:space:]]*"\([^"]*\)".*/\1/' | head -1)
    
    if [[ -z "$value" ]]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# 列出所有实验
list_experiments() {
    local status_filter="$1"
    
    if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
        print_error "实验目录不存在: $EXPERIMENTS_DIR"
        return 1
    fi
    
    local experiments=()
    for exp_dir in "$EXPERIMENTS_DIR"/*; do
        if [[ -d "$exp_dir" ]]; then
            local exp_id=$(basename "$exp_dir")
            local info_file="$exp_dir/experiment_info.json"
            
            if [[ -f "$info_file" ]]; then
                local status=$(extract_json_value "$info_file" "status" "unknown")
                local created_at=$(extract_json_value "$info_file" "created_at" "")
                local description=$(extract_json_value "$info_file" "description" "")
                
                if [[ -z "$status_filter" || "$status" == "$status_filter" ]]; then
                    experiments+=("$exp_id|$status|$created_at|$description")
                fi
            fi
        fi
    done
    
    if [[ ${#experiments[@]} -eq 0 ]]; then
        print_info "没有找到任何实验"
        return 0
    fi
    
    # 按创建时间排序
    IFS=$'\n' experiments=($(printf '%s\n' "${experiments[@]}" | sort -t'|' -k3 -r))
    
    echo -e "\n找到 ${#experiments[@]} 个实验:"
    echo "--------------------------------------------------------------------------------"
    printf "%-30s %-10s %-20s %-20s\n" "ID" "状态" "创建时间" "描述"
    echo "--------------------------------------------------------------------------------"
    
    for exp in "${experiments[@]}"; do
        IFS='|' read -r exp_id status created_at description <<< "$exp"
        local short_desc="${description:0:20}"
        [[ ${#description} -gt 20 ]] && short_desc="${short_desc}..."
        local short_time="${created_at:0:19}"
        printf "%-30s %-10s %-20s %-20s\n" "$exp_id" "$status" "$short_time" "$short_desc"
    done
    
    echo "--------------------------------------------------------------------------------"
}

# 显示实验详情
show_experiment() {
    local experiment_id="$1"
    local exp_dir="$EXPERIMENTS_DIR/$experiment_id"
    
    if [[ ! -d "$exp_dir" ]]; then
        print_error "实验不存在: $experiment_id"
        return 1
    fi
    
    local info_file="$exp_dir/experiment_info.json"
    if [[ ! -f "$info_file" ]]; then
        print_error "实验信息文件不存在: $info_file"
        return 1
    fi
    
    echo -e "\n实验详情: $experiment_id"
    echo "--------------------------------------------------"
    
    # 基本信息
    local name=$(extract_json_value "$info_file" "name" "")
    local description=$(extract_json_value "$info_file" "description" "")
    local status=$(extract_json_value "$info_file" "status" "")
    local created_at=$(extract_json_value "$info_file" "created_at" "")
    local config_hash=$(extract_json_value "$info_file" "config_hash" "")
    
    echo "名称: $name"
    echo "描述: $description"
    echo "状态: $status"
    echo "创建时间: $created_at"
    echo "配置哈希: $config_hash"
    
    # 目录结构
    echo -e "\n目录结构:"
    echo "  根目录: $exp_dir"
    echo "  模型: $exp_dir/models"
    echo "  日志: $exp_dir/logs"
    echo "  图片: $exp_dir/figures"
    echo "  Checkpoints: $exp_dir/checkpoints"
    
    # 文件统计
    echo -e "\n文件统计:"
    for name in "模型文件" "日志文件" "图片文件" "Checkpoint文件"; do
        case "$name" in
            "模型文件") path="$exp_dir/models" ;;
            "日志文件") path="$exp_dir/logs" ;;
            "图片文件") path="$exp_dir/figures" ;;
            "Checkpoint文件") path="$exp_dir/checkpoints" ;;
        esac
        
        if [[ -d "$path" ]]; then
            local count=$(find "$path" -type f 2>/dev/null | wc -l)
            echo "  $name: $count 个文件"
        else
            echo "  $name: 目录不存在"
        fi
    done
}

# 清理旧实验
clean_experiments() {
    local keep_latest="$1"
    local status_filter="$2"
    local dry_run="$3"
    
    if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
        print_error "实验目录不存在: $EXPERIMENTS_DIR"
        return 1
    fi
    
    # 获取所有实验
    local experiments=()
    for exp_dir in "$EXPERIMENTS_DIR"/*; do
        if [[ -d "$exp_dir" ]]; then
            local exp_id=$(basename "$exp_dir")
            local info_file="$exp_dir/experiment_info.json"
            
            if [[ -f "$info_file" ]]; then
                local status=$(extract_json_value "$info_file" "status" "unknown")
                local created_at=$(extract_json_value "$info_file" "created_at" "")
                
                if [[ -z "$status_filter" || "$status" == "$status_filter" ]]; then
                    experiments+=("$exp_id|$status|$created_at")
                fi
            fi
        fi
    done
    
    # 按创建时间排序
    IFS=$'\n' experiments=($(printf '%s\n' "${experiments[@]}" | sort -t'|' -k3 -r))
    
    if [[ ${#experiments[@]} -le $keep_latest ]]; then
        print_info "只有 ${#experiments[@]} 个实验，无需清理"
        return 0
    fi
    
    local to_delete=("${experiments[@]:$keep_latest}")
    
    echo -e "\n将删除 ${#to_delete[@]} 个旧实验:"
    for exp in "${to_delete[@]}"; do
        IFS='|' read -r exp_id status created_at <<< "$exp"
        echo "  - $exp_id ($status)"
    done
    
    if [[ "$dry_run" == "true" ]]; then
        print_warning "[DRY RUN] 以上实验将被删除"
        return 0
    fi
    
    echo -n -e "\n确认删除这 ${#to_delete[@]} 个实验？(y/N): "
    read -r confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_info "取消删除"
        return 0
    fi
    
    local deleted_count=0
    for exp in "${to_delete[@]}"; do
        IFS='|' read -r exp_id status created_at <<< "$exp"
        local exp_path="$EXPERIMENTS_DIR/$exp_id"
        
        if [[ -d "$exp_path" ]]; then
            if rm -rf "$exp_path"; then
                ((deleted_count++))
                print_success "已删除: $exp_id"
            else
                print_error "删除失败: $exp_id"
            fi
        fi
    done
    
    print_success "成功删除 $deleted_count 个实验"
}

# 迁移旧输出结构
migrate_old_outputs() {
    local dry_run="$1"
    
    if [[ ! -d "$OLD_OUTPUTS_DIR" ]]; then
        print_error "旧输出目录不存在: $OLD_OUTPUTS_DIR"
        return 1
    fi
    
    print_info "扫描旧输出目录: $OLD_OUTPUTS_DIR"
    
    # 查找所有lora目录
    local lora_dirs=()
    local lora_path="$OLD_OUTPUTS_DIR/lora"
    
    if [[ -d "$lora_path" ]]; then
        while IFS= read -r -d '' lora_dir; do
            lora_dirs+=("$lora_dir")
        done < <(find "$lora_path" -maxdepth 1 -type d -print0 | tail -n +2)
    fi
    
    if [[ ${#lora_dirs[@]} -eq 0 ]]; then
        print_info "没有找到旧的LoRA目录"
        return 0
    fi
    
    print_info "找到 ${#lora_dirs[@]} 个LoRA目录"
    
    # 创建实验目录
    mkdir -p "$EXPERIMENTS_DIR"
    
    local migrated_count=0
    for lora_dir in "${lora_dirs[@]}"; do
        local lora_name=$(basename "$lora_dir")
        local exp_name="migrated_$lora_name"
        local exp_dir="$EXPERIMENTS_DIR/$exp_name"
        
        if [[ "$dry_run" == "true" ]]; then
            print_warning "[DRY RUN] 将迁移: $lora_name -> $exp_name"
            continue
        fi
        
        # 创建实验目录结构
        mkdir -p "$exp_dir"/{models/{lora,processor},logs,figures,checkpoints,config}
        
        # 创建实验信息文件
        cat > "$exp_dir/experiment_info.json" << EOF
{
  "experiment_id": "$exp_name",
  "name": "$exp_name",
  "description": "从旧输出迁移: $lora_name",
  "created_at": "$(date -Iseconds)",
  "config_hash": "",
  "status": "completed",
  "metrics": null
}
EOF
        
        # 复制LoRA文件
        if [[ -f "$lora_dir/adapter_model.safetensors" ]]; then
            cp -r "$lora_dir"/* "$exp_dir/models/lora/"
        fi
        
        # 复制Processor文件
        local proc_dir="$OLD_OUTPUTS_DIR/processor/$lora_name"
        if [[ -d "$proc_dir" ]]; then
            cp -r "$proc_dir"/* "$exp_dir/models/processor/"
        fi
        
        # 复制日志文件
        local logs_dir="$OLD_OUTPUTS_DIR/logs"
        if [[ -d "$logs_dir" ]]; then
            find "$logs_dir" -name "*$lora_name*" -exec cp {} "$exp_dir/logs/" \;
        fi
        
        # 复制图片文件
        local figures_dir="$OLD_OUTPUTS_DIR/figures"
        if [[ -d "$figures_dir" ]]; then
            find "$figures_dir" -name "*$lora_name*" -exec cp {} "$exp_dir/figures/" \;
        fi
        
        ((migrated_count++))
        print_success "已迁移: $lora_name -> $exp_name"
    done
    
    if [[ "$dry_run" != "true" ]]; then
        print_success "成功迁移 $migrated_count 个实验"
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
实验管理工具 (bash版本)

用法: $0 <command> [options]

命令:
  list [--status STATUS]                   列出所有实验
  show <experiment_id>                      显示实验详情
  clean [--keep-latest N] [--status STATUS] [--dry-run]  清理旧实验
  migrate [--dry-run]                       迁移旧输出结构

选项:
  --experiments-dir DIR                     实验目录 (默认: experiments)
  --old-outputs-dir DIR                    旧输出目录 (默认: outputs)
  --keep-latest N                          保留最新N个实验 (默认: 5)
  --status STATUS                          按状态过滤 (running|completed|failed)
  --dry-run                                预览模式，不实际执行

环境变量:
  EXPERIMENTS_DIR                          实验目录
  OLD_OUTPUTS_DIR                          旧输出目录

示例:
  $0 list
  $0 list --status completed
  $0 show vl_lora_train_20250101_120000
  $0 clean --keep-latest 3 --dry-run
  $0 migrate --dry-run
EOF
}

# 主函数
main() {
    # 检查是否是帮助选项
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command="$1"
    shift || true
    
    # 解析全局选项
    while [[ $# -gt 0 ]]; do
        case $1 in
            --experiments-dir)
                EXPERIMENTS_DIR="$2"
                shift 2
                ;;
            --old-outputs-dir)
                OLD_OUTPUTS_DIR="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    case "$command" in
        list)
            local status_filter=""
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --status)
                        status_filter="$2"
                        shift 2
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            list_experiments "$status_filter"
            ;;
        show)
            if [[ $# -eq 0 ]]; then
                print_error "请提供实验ID"
                exit 1
            fi
            show_experiment "$1"
            ;;
        clean)
            local keep_latest=5
            local status_filter=""
            local dry_run="false"
            
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --keep-latest)
                        keep_latest="$2"
                        shift 2
                        ;;
                    --status)
                        status_filter="$2"
                        shift 2
                        ;;
                    --dry-run)
                        dry_run="true"
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            clean_experiments "$keep_latest" "$status_filter" "$dry_run"
            ;;
        migrate)
            local dry_run="false"
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --dry-run)
                        dry_run="true"
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            migrate_old_outputs "$dry_run"
            ;;
        *)
            print_error "未知命令: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
