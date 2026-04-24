<#
.SYNOPSIS
    Restormer Encoder 全量可视化批处理脚本

.DESCRIPTION
    对 MDTA/HTA/WTA 三种注意力机制的 Encoder Level 1-4 (含 Latent) 所有 Block
    批量调用 visualize_attention.py 生成可视化图。
    在脚本内通过 $LEVEL_CONFIG 控制要可视化的层级和 Block 数量。

.EXAMPLE
    # 在项目根目录运行
    .\scripts\visualization\visualize_encoder_all.ps1

.NOTES
    运行前需确认:
    - $MODELS 中的权重文件路径存在
    - $NORMAL_IMAGE / $LOW_IMAGE 图像文件存在
    - 输出目录: visualization\encoder\
#>

$ErrorActionPreference = "Stop"

# 配置路径
$PROJECT_ROOT = "D:\Workspace\A_Projects\Thesis\LLIE\Restormer_LLIE"
$EXPERIMENTS_DIR = "$PROJECT_ROOT\experiments"
$OUTPUT_DIR = "$PROJECT_ROOT\visualization\encoder"
$SCRIPT_PATH = "$PROJECT_ROOT\scripts\visualization\visualize_attention.py"

# 图像路径
$NORMAL_IMAGE = "$PROJECT_ROOT\normal00323.png"
$LOW_IMAGE = "$PROJECT_ROOT\low00323.png"

# 模型权重路径
$MODELS = @{
    "HTA"  = "$EXPERIMENTS_DIR\Restormer_128_2_60k_HTA\net_g_44000.pth"
    "WTA"  = "$EXPERIMENTS_DIR\Restormer_128_2_60k_WTA\net_g_44000.pth"
    "MDTA" = "$EXPERIMENTS_DIR\Restormer_128_2_60k_MDTA\net_g_44000.pth"
}

# 层配置: level -> num_blocks
$LEVEL_CONFIG = @{
    1 = 4   # encoder_level1: 4 blocks
    # 2 = 6   # encoder_level2: 6 blocks
    # 3 = 6   # encoder_level3: 6 blocks
    # 4 = 8   # latent: 8 blocks
}

# 图像列表
$IMAGES = @($NORMAL_IMAGE, $LOW_IMAGE)

# 图像尺寸
$IMAGE_SIZE = 128

Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Restormer Encoder 全量可视化" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "输出目录: $OUTPUT_DIR"
Write-Host "图像尺寸: $IMAGE_SIZE"
Write-Host ""

# 检查文件是否存在
foreach ($img in $IMAGES) {
    if (-not (Test-Path $img)) {
        Write-Host "错误: 图像不存在 - $img" -ForegroundColor Red
        exit 1
    }
}

foreach ($attn_type in $MODELS.Keys) {
    $checkpoint = $MODELS[$attn_type]
    if (-not (Test-Path $checkpoint)) {
        Write-Host "警告: 模型权重不存在 - $checkpoint" -ForegroundColor Yellow
        continue
    }
}

# 切换到项目目录
Set-Location $PROJECT_ROOT

# 计算总任务数
$total_tasks = $MODELS.Count * $IMAGES.Count * ($LEVEL_CONFIG.Values | Measure-Object -Sum).Sum
$current_task = 0

Write-Host "总任务数: $total_tasks" -ForegroundColor Green
Write-Host ""

# 遍历所有组合
foreach ($attn_type in $MODELS.Keys) {
    $checkpoint = $MODELS[$attn_type]
    
    if (-not (Test-Path $checkpoint)) {
        Write-Host "跳过 $attn_type (权重不存在)" -ForegroundColor Yellow
        continue
    }
    
    Write-Host ""
    Write-Host ("=" * 50) -ForegroundColor Magenta
    Write-Host "处理注意力类型: $attn_type" -ForegroundColor Magenta
    Write-Host ("=" * 50) -ForegroundColor Magenta
    
    foreach ($image_path in $IMAGES) {
        $image_name = [System.IO.Path]::GetFileNameWithoutExtension($image_path)
        
        Write-Host ""
        Write-Host ("-" * 40) -ForegroundColor Blue
        Write-Host "图像: $image_name" -ForegroundColor Blue
        Write-Host ("-" * 40) -ForegroundColor Blue
        
        foreach ($level in $LEVEL_CONFIG.Keys | Sort-Object) {
            $num_blocks = $LEVEL_CONFIG[$level]
            $level_name = if ($level -eq 4) { "latent" } else { "encoder$level" }
            
            foreach ($block in 0..($num_blocks - 1)) {
                $current_task++
                $progress = [math]::Round(($current_task / $total_tasks) * 100, 1)
                
                Write-Host "[$current_task/$total_tasks] ($progress%) $attn_type - $image_name - $level_name - block$block" -ForegroundColor White
                
                # 构建命令
                $cmd = @(
                    "uv", "run", "python", $SCRIPT_PATH,
                    "--image", $image_path,
                    "--checkpoint", $checkpoint,
                    "--attn_type", $attn_type,
                    "--level", $level,
                    "--block", $block,
                    "--size", $IMAGE_SIZE,
                    "--output_dir", $OUTPUT_DIR
                )
                
                # 执行命令
                try {
                    & $cmd[0] $cmd[1..($cmd.Length-1)] 2>&1 | Out-Null
                    Write-Host "  完成" -ForegroundColor Green
                }
                catch {
                    Write-Host "  失败: $_" -ForegroundColor Red
                }
            }
        }
    }
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "可视化完成!" -ForegroundColor Cyan
Write-Host "结果保存在: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
