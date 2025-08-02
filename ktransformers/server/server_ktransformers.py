import os
import re
import threading
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn.logging
import uvicorn
import sys
import atexit
import torch
import gc
import logging
import traceback
import subprocess
import signal
import psutil
import asyncio
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
from fastapi.middleware.cors import CORSMiddleware
from ktransformers.server.args import ArgumentParser
from ktransformers.server.config.config import Config
from ktransformers.server.utils.create_interface import create_interface, GlobalInterface
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ktransformers.server.api import router, post_db_creation_operations
from ktransformers.server.utils.sql_utils import Base, SQLUtil
from ktransformers.server.config.log import logger

# Create logs directory if it doesn't exist
log_dir = os.path.join(project_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "server.log")

# Remove all existing handlers from root logger
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create formatters
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_formatter = logging.Formatter("%(levelname)s - %(message)s")

# Create handlers
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

# Configure root logger
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Configure specific loggers
loggers_to_configure = ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "httpx", "ktransformers"]

for logger_name in loggers_to_configure:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Don't propagate to root logger to avoid duplicate logs
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Test logging
logging.info("=== Server Starting ===")
logging.debug("Debug logging enabled")
logging.info("Info logging enabled")
logging.warning("Warning logging enabled")
logging.error("Error logging enabled")

# Global variable for the subprocess
model_process = None


def write_log(message, level="INFO"):
    """Write log message directly to file and console."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {level} - {message}\n"

        # Write to console
        print(log_message, end="")

        # Write to log file
        log_file = "/models/ktransformers/logs/server.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message)
    except Exception as e:
        print(f"Error writing to log: {str(e)}")


def rotate_log_file(log_file, max_size_mb=10):
    """Rotate log file if it exceeds the maximum size.

    Args:
        log_file (str): Path to the log file
        max_size_mb (int): Maximum size in MB before rotation
    """
    try:
        if not os.path.exists(log_file):
            return

        # Get file size in MB
        file_size_mb = os.path.getsize(log_file) / (1024 * 1024)

        if file_size_mb > max_size_mb:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{log_file}.{timestamp}"

            # Copy current log file to backup
            import shutil

            shutil.copy2(log_file, backup_file)

            # Empty the current log file
            with open(log_file, "w") as f:
                f.write(f"=== Log rotated at {datetime.now()} ===\n")

            write_log(f"Log file rotated. Backup created at: {backup_file}")
    except Exception as e:
        write_log(f"Error rotating log file: {str(e)}", "ERROR")


def start_model_server():
    """Start the model server as a subprocess."""
    global model_process
    try:
        if model_process is not None:
            write_log("Model server is already running", "WARNING")
            return False

        # Check and rotate log file before starting
        log_file = "/models/ktransformers/logs/server.log"
        rotate_log_file(log_file)

        # Construct the command
        cmd = [
            "cd",
            "/models/ktransformers/ktransformers-deepseek-r1-0528/ktransformers",
            "&&",
            "python",
            "-u",
            "-m",
            "ktransformers.server.main",  # Add -u for unbuffered output
            "--gguf_path",
            "/models/deepseek/r1-0528/gguf/Q4_K_M",
            "--model_path",
            "/models/deepseek/r1-0528/config",
            "--max_new_tokens",
            "10240",
            "--cpu_infer",
            "150",
            "--cache_lens",
            "10240",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
            "2>&1",
            "|",
            "stdbuf",
            "-oL",
            "tee",
            "-a",
            "/models/ktransformers/logs/server.log",  # Use tee to write to both stdout and log file
        ]

        # Start the subprocess with shell=True to handle the cd command
        write_log(f"Starting model server with command: {' '.join(cmd)}")

        # Create a process that will capture output and write to log file
        process = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # Line buffered
        )

        # Store the process
        model_process = process

        # Check if process started successfully
        if model_process.poll() is not None:
            write_log("Failed to start model server", "ERROR")
            model_process = None
            return False

        write_log(f"Model server started with PID: {model_process.pid}")

        # Wait for the specific loading message
        target_message = "loading model.norm.weight to cuda:0"
        max_wait_time = 300  # 5 minutes timeout
        start_time = datetime.now()
        last_rotation_check = start_time

        while (datetime.now() - start_time).total_seconds() < max_wait_time:
            # Check log rotation every minute
            current_time = datetime.now()
            if (current_time - last_rotation_check).total_seconds() >= 60:
                rotate_log_file(log_file)
                last_rotation_check = current_time

            # Read a line from the process output
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                # Process has ended
                break

            if line:
                # Log the output
                write_log(line.strip())

                # Check for target message
                if target_message in line:
                    # Wait additional 3 seconds after finding the message
                    import time

                    time.sleep(3)
                    return True

        write_log("Timeout waiting for model loading message", "ERROR")
        return False

    except Exception as e:
        write_log(f"Error starting model server: {str(e)}", "ERROR")
        write_log(traceback.format_exc(), "ERROR")
        model_process = None
        return False


def stop_model_server():
    """Stop the model server subprocess."""
    global model_process
    try:
        if model_process is None:
            logger.warning("No model server is running")
            return False

        # Get the process and all its children
        parent = psutil.Process(model_process.pid)
        children = parent.children(recursive=True)

        # Terminate all child processes
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate the parent process
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

        # Wait for processes to terminate
        gone, alive = psutil.wait_procs([parent] + children, timeout=3)

        # Force kill if still alive
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

        model_process = None
        logger.info("Model server stopped successfully")
        return True

    except Exception as e:
        logger.error(f"Error stopping model server: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def mount_app_routes(mount_app: FastAPI):
    sql_util = SQLUtil()
    logger.info("Creating SQL tables")
    Base.metadata.create_all(bind=sql_util.sqlalchemy_engine)
    post_db_creation_operations()
    mount_app.include_router(router)

    # Add model management endpoints
    @mount_app.post("/load_model")
    async def load_model_endpoint():
        """Start the model server."""
        try:
            # Check if model is already running
            if model_process is not None and model_process.poll() is None:
                logger.info("Model server is already running")
                print("Model server is already running")
                return {"status": "success", "message": "Model server is already running"}

            if start_model_server():
                return {"status": "success", "message": "Model server started successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to start model server")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @mount_app.post("/unload_model")
    async def unload_model_endpoint():
        """Stop the model server."""
        try:
            if model_process is None:
                return {"status": "success", "message": "Model server is already unloaded"}

            if stop_model_server():
                return {"status": "success", "message": "Model server stopped successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to stop model server")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @mount_app.get("/model_info")
    async def get_model_info():
        """Get information about the model server status."""
        try:
            if model_process is None or model_process.poll() is not None:
                return {"status": "Model server not running", "model_type": "deepseek-coder"}

            return {"status": "Model server running", "model_type": "deepseek-coder", "pid": model_process.pid}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @mount_app.post("/self-shutdown")
    async def self_shutdown():
        def exit_later():
            # short delay to ensure response is sent
            import time

            time.sleep(0.1)
            os._exit(0)  # immediate hard exit

        threading.Thread(target=exit_later, daemon=True).start()
        return {"status": "shutting down"}


def create_app():
    cfg = Config()
    arg_parser = ArgumentParser(cfg)
    args = arg_parser.parse_args()

    # Create app with increased timeout
    app = FastAPI()

    # Add timeout middleware
    @app.middleware("http")
    async def timeout_middleware(request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=1200)  # 20 minutes
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")

    if Config().web_cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app)
    if cfg.mount_web:
        mount_index_routes(app)
    return app


def update_web_port(config_file: str):
    ip_port_pattern = (
        r"(localhost|((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)):[0-9]{1,5}"
    )
    with open(config_file, "r", encoding="utf-8") as f_cfg:
        web_config = f_cfg.read()
    ip_port = "localhost:" + str(Config().server_port)
    new_web_config = re.sub(ip_port_pattern, ip_port, web_config)
    with open(config_file, "w", encoding="utf-8") as f_cfg:
        f_cfg.write(new_web_config)


def mount_index_routes(app: FastAPI):
    project_dir = os.path.dirname(os.path.dirname(__file__))
    web_dir = os.path.join(project_dir, "website/dist")
    web_config_file = os.path.join(web_dir, "config.js")
    update_web_port(web_config_file)
    if os.path.exists(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir), name="static")
    else:
        err_str = f"No website resources in {web_dir}, please complile the website by npm first"
        logger.error(err_str)
        print(err_str)
        exit(1)


def run_api(app, host, port, **kwargs):
    # Configure logging to write to both console and file
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "server.log")

    # Configure uvicorn logging
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": log_file,
                "mode": "a",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console", "file"], "level": "DEBUG"},
            "uvicorn.error": {"handlers": ["console", "file"], "level": "DEBUG"},
            "uvicorn.access": {"handlers": ["console", "file"], "level": "DEBUG"},
            "fastapi": {"handlers": ["console", "file"], "level": "DEBUG"},
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
    }

    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
            log_config=log_config,
            log_level="debug",
        )
    else:
        uvicorn.run(app, host=host, port=port, log_config=log_config, log_level="debug")


def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ktransformers server",
        version="1.0.0",
        summary="This is a server that provides a RESTful API for ktransformers.",
        description="We provided chat completion and openai assistant interfaces.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {"url": "https://kvcache.ai/media/icon_1.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def cleanup():
    """Cleanup function to ensure the model server is stopped when the main server exits."""
    if model_process is not None:
        stop_model_server()


def main():
    # Register cleanup function
    atexit.register(cleanup)

    # Test file writing explicitly
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "server.log")

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Server Starting at {datetime.now()} ===\n")
            f.write("Testing file write access...\n")
            f.write(f"Log file path: {log_file}\n")
            f.write(f"Current working directory: {os.getcwd()}\n")
            f.write(f"Project directory: {project_dir}\n")
            f.write("=== End of test message ===\n")
        print(f"Successfully wrote to log file: {log_file}")
    except Exception as e:
        print(f"Failed to write to log file: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Project directory: {project_dir}")
        print(f"Log directory: {log_dir}")
        print(f"Log file path: {log_file}")
        print(f"Directory exists: {os.path.exists(log_dir)}")
        print(f"Directory writable: {os.access(log_dir, os.W_OK)}")
        print(f"File exists: {os.path.exists(log_file)}")
        print(f"File writable: {os.access(log_file, os.W_OK) if os.path.exists(log_file) else 'N/A'}")

    # Create app
    app = create_app()
    custom_openapi(app)

    # Get configuration
    cfg = Config()
    arg_parser = ArgumentParser(cfg)
    args = arg_parser.parse_args()

    # Run the API server
    run_api(
        app=app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
