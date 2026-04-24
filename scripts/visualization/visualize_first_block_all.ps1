<#
.SYNOPSIS
    批量可视化 Encoder Level 1 Block 0 的完整数据流

.DESCRIPTION
    对多张图像 × MDTA/HTA/WTA 三种注意力机制，批量调用 visualize_first_block.py，
    生成 Patch Embedding → Q/K/V → Attention Map → Output 的完整可视化。
    在脚本内通过 $IMAGES 和 $CHECKPOINTS 配置图像和权重。

.EXAMPLE
    # 在项目根目录运行
    .\scripts\visualization\visualize_first_block_all.ps1

.NOTES
    运行前需确认:
    - $CHECKPOINTS 中的权重文件路径存在
    - $IMAGES 中的图像文件位于项目根目录
    - 输出目录: visualization\first_block\
#>

$IMAGES = @(
    "low00729.png",
    "low00323.png",
    "low00736.png"
)
$EXPERIMENTS_DIR = "experiments"
$OUTPUT_DIR = "visualization\first_block"

# 三种注意力机制的 checkpoint 路径
$CHECKPOINTS = @{
    "HTA"  = "$EXPERIMENTS_DIR\Restormer_128_2_60k_HTA\net_g_44000.pth"
    "WTA"  = "$EXPERIMENTS_DIR\Restormer_128_2_60k_WTA\net_g_44000.pth"
    "MDTA" = "$EXPERIMENTS_DIR\Restormer_128_2_60k_MDTA\net_g_44000.pth"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Visualizing First Block (Encoder Level 1 Block 0)" -ForegroundColor Cyan
Write-Host "Images: $($IMAGES.Count) files" -ForegroundColor Yellow
Write-Host "Output: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

foreach ($IMAGE in $IMAGES) {
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host "Processing Image: $IMAGE" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host ""
    
    foreach ($attn_type in @("MDTA", "HTA", "WTA")) {
        $checkpoint = $CHECKPOINTS[$attn_type]
        
        Write-Host "  Processing: $attn_type" -ForegroundColor Green
        Write-Host "  Checkpoint: $checkpoint" -ForegroundColor Gray
        
        if (-not (Test-Path $checkpoint)) {
            Write-Host "    [WARNING] Checkpoint not found: $checkpoint" -ForegroundColor Red
            continue
        }
        
        # 运行可视化
        python scripts/visualization/visualize_first_block.py `
            --image $IMAGE `
            --checkpoint $checkpoint `
            --attn_type $attn_type `
            --no_resize `
            --output_dir $OUTPUT_DIR
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    [SUCCESS] $attn_type completed" -ForegroundColor Green
        } else {
            Write-Host "    [ERROR] $attn_type failed" -ForegroundColor Red
        }
        Write-Host ""
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All visualizations completed!" -ForegroundColor Green
Write-Host "Results saved to: $OUTPUT_DIR\" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
