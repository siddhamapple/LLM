import argparse
from src.config_manager import load_config
from src.logger import get_logger

def main():
    parser = argparse.ArgumentParser(description="LLM Development & Fine-Tuning")
    parser.add_argument("--mode", choices=["pretrain", "finetune", "infer"], required=True, help="Which pipeline to run")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        name="app",
        log_dir=cfg.paths.logs_dir,
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        file_name=cfg.logging.file_name,
    )
    logger.info(f"Project: {cfg.project.name}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "pretrain":
        from src.pipelines.pretrain_pipeline import run as run_pretrain
        run_pretrain(cfg, logger)
    elif args.mode == "finetune":
        from src.pipelines.finetune_pipeline import run as run_finetune
        run_finetune(cfg, logger)
    elif args.mode == "infer":
        from src.pipelines.inference_pipeline import run as run_infer
        run_infer(cfg, logger)

if __name__ == "__main__":
    main()
