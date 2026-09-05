"""
Main Application Entrypoint.

Provides CLI command execution and options to launch pipeline monitoring or Tkinter GUI.
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure package modules can be imported correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigManager, setup_logger, LOGS_DIR

logger = setup_logger(log_file=LOGS_DIR / "app.log")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Based Real-Time Wheel Alignment Monitoring and Alert System Using Computer Vision"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to input video file or camera device index (e.g., 0)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to custom settings YAML configuration file"
    )
    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        help="Launch desktop Tkinter GUI Dashboard"
    )
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Launch browser-based FastAPI Localhost Web Dashboard"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port number for Web Application Server (default: 8000)"
    )
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Display live OpenCV rendering window during execution"
    )
    parser.add_argument(
        "--debug", "-dbg",
        action="store_true",
        help="Enable pipeline debug mode with performance HUD overlays and ROI crop previews"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run startup environment and dependency validation check"
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    if args.validate:
        from scripts.validate_environment import run_full_validation
        run_full_validation()
        return

    # Initialize configuration
    config_path = Path(args.config) if args.config else None
    config_manager = ConfigManager(config_path=config_path)

    logger.info("Initializing Wheel Alignment AI Monitoring System...")

    if args.web:
        try:
            logger.info(f"Launching FastAPI Localhost Web Dashboard on http://127.0.0.1:{args.port}...")
            from src.web_app import run_web_server
            run_web_server(host="127.0.0.1", port=args.port)
        except Exception as e:
            logger.critical(f"Failed to start FastAPI Web Application Server: {e}")
            sys.exit(1)
    elif args.gui:
        try:
            logger.info("Launching Tkinter Desktop GUI Dashboard...")
            from gui.dashboard import launch_gui
            launch_gui()
        except Exception as e:
            logger.critical(f"Failed to start Tkinter GUI: {e}")
            sys.exit(1)
    else:
        try:
            from src.pipeline import WheelAlignmentPipeline
        except ImportError as e:
            logger.error(f"Missing required dependency to run pipeline: {e}")
            logger.info("Please install dependencies using: pip install -r requirements.txt")
            sys.exit(1)

        video_src = args.video if args.video is not None else config_manager.get("video.source", 0)
        pipeline = WheelAlignmentPipeline(config_manager=config_manager, debug_mode=args.debug)

        try:
            logger.info(f"Running monitoring pipeline on source: {video_src} (Debug: {args.debug})")
            # Automatically enable display when debug flag is specified unless user suppressed it
            display_on = args.display or args.debug
            pipeline.run_video(video_source=video_src, display=display_on)
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user (KeyboardInterrupt).")
            pipeline.stop()
        except Exception as e:
            logger.error(f"Execution error in main pipeline loop: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
