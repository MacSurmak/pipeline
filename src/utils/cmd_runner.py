"""
DRY utility for executing subprocess commands with logging, error handling, 
and real-time log parsing for rich progress bars.
"""
import subprocess
import time
import re
import shutil
from pathlib import Path
from loguru import logger

def run_command(cmd: list, cwd: Path, log_name: str, progress=None, task_id=None) -> bool:
    """
    Executes a shell command, logs stdout/stderr to a file, handles errors,
    and isolates all log files into a dedicated logs/ subdirectory.
    """
    cmd_str = " ".join(str(x) for x in cmd)
    
    # Isolate all logs
    log_dir = cwd / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    
    logger.info(f"Executing: {cmd[0]} (Logging to {log_file.name})")
    logger.debug(f"Full command: {cmd_str}")
    
    start_time = time.time()
    
    lig_pattern = re.compile(r"^LIG\s+(\d+)")
    
    # Identify target Job Control log (e.g. dock_f1.log) for real-time progress parsing
    target_job_log = None
    if len(cmd) > 1 and str(cmd[1]).endswith(".in"):
        target_job_log = cwd / f"{Path(cmd[1]).stem}.log"

    # Remove old log from previous runs to prevent 180/180 glitch
    if target_job_log:
        target_job_log.unlink(missing_ok=True)

    try:
        # Direct stdout to log_file to prevent 64kb pipe deadlocks
        with open(log_file, "w") as f_out:
            f_out.write(f"COMMAND: {cmd_str}\n\nOUTPUT:\n")
            f_out.flush()
            
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=f_out,
                stderr=subprocess.STDOUT
            )

            last_pos = 0
            while process.poll() is None:
                # Read target job log for real-time Glide progress
                if progress and task_id and target_job_log and target_job_log.exists():
                    with open(target_job_log, "r", errors="ignore") as f_job:
                        f_job.seek(last_pos)
                        for line in f_job:
                            match = lig_pattern.match(line)
                            if match:
                                progress.update(task_id, completed=int(match.group(1)))
                        last_pos = f_job.tell()
                time.sleep(0.5)

        returncode = process.returncode
        
        # Guarantee 100% completion for the progress bar
        if progress and task_id and returncode == 0:
            task = progress._tasks.get(task_id)
            if task and task.total:
                progress.update(task_id, completed=task.total)

        # Automatically move all stray .log files (created by Job Control) to logs/
        for stray_log in cwd.glob("*.log"):
            if stray_log.is_file():
                shutil.move(str(stray_log), str(log_dir / stray_log.name))

    except Exception as e:
        logger.critical(f"Failed to launch command {cmd[0]}: {e}")
        return False

    duration = time.time() - start_time
    
    if returncode != 0:
        logger.error(f"Command '{cmd[0]}' failed with exit code {returncode}.")
        # Read the tail of the log file for quick error debugging
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                last_lines = "".join(lines[-10:])
                logger.error(f"Tail of output:\n{last_lines}")
        except Exception:
            pass
        return False
        
    logger.success(f"Command '{cmd[0]}' completed successfully in {duration:.1f}s.")
    return True