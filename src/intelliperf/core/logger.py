################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional


class Logger:
    """
    Simple buffered logger for optimization runs and interactions.
    Records events in memory and provides flush capability for persistence.
    """
    
    def __init__(self, run_name: str = None):
        self.buffer: List[Dict[str, Any]] = []
        self.run_id = str(uuid.uuid4())
        self.run_name = run_name or f"run_{int(time.time())}"
        self.start_time = time.time()
        
        # Record run start
        self.record("run_start", {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "timestamp": self.start_time
        })
    
    def record(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record an event - always succeeds, just adds to buffer.
        
        Args:
            event_type: Type of event (e.g., "llm_call", "optimization_pass", "validation_result")
            data: Event data dictionary
        """
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        }
        self.buffer.append(entry)
        
        # Also log to console for immediate visibility
        logging.debug(f"Logger: {event_type} - {data}")
    
    def get_buffer(self) -> List[Dict[str, Any]]:
        """
        Get current buffer - for debugging/fallback.
        
        Returns:
            Copy of current buffer
        """
        return self.buffer.copy()
    
    def get_run_summary(self) -> Dict[str, Any]:
        """
        Get summary of the current run.
        
        Returns:
            Dictionary with run summary
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Count different event types
        event_counts = {}
        for entry in self.buffer:
            event_type = entry["type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "start_time": self.start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "total_events": len(self.buffer),
            "event_counts": event_counts,
            "events": self.buffer
        }
    
    def flush(self, output_file: Optional[str] = None) -> bool:
        """
        Flush logs to output targets with error handling.
        
        Args:
            output_file: Optional file path for JSON output
            
        Returns:
            True if flush was successful, False otherwise
        """
        success = True
        run_summary = self.get_run_summary()
        
        # Always try console output first (never fails)
        try:
            logging.info("=== Run Summary ===")
            logging.info(f"Run ID: {run_summary['run_id']}")
            logging.info(f"Run Name: {run_summary['run_name']}")
            logging.info(f"Duration: {run_summary['duration_seconds']:.2f}s")
            logging.info(f"Total Events: {run_summary['total_events']}")
            logging.info("Event Counts:")
            for event_type, count in run_summary['event_counts'].items():
                logging.info(f"  {event_type}: {count}")
            logging.info("==================")
        except Exception as e:
            logging.error(f"Console flush failed: {e}")
            success = False
        
        # Try file output if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(run_summary, f, indent=2)
                logging.info(f"Logs flushed to: {output_file}")
            except Exception as e:
                logging.error(f"File flush failed: {e}")
                success = False
        
        return success 
