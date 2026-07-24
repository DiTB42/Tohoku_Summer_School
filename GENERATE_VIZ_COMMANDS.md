uv run python viz_checkpoints.py models/checkpoints/sac_snake_1000000_steps.zip --seed 1 --record out.mp4 --video-seconds 60 --video-fps 30 --video-res 1280x1280

uv run python benchmark_models.py --models-dir models/checkpoints_no_terrain --no-terrain --n-mazes 30 --jobs 8 --pick 1000,4000,6000,10000,16000,24000,50000,74000,100000,250000,500000,750000,1000000

uv run python benchmark_models.py --models-dir models/checkpoints --n-mazes 30 --jobs 8 --pick 1000,4000,6000,10000,16000,24000,50000,74000,100000,250000,500000,750000,1000000,1500000