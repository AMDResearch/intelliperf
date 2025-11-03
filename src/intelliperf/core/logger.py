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
import os
import time
import uuid
from datetime import datetime
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
		self.record(
			"run_start",
			{
				"run_id": self.run_id,
				"run_name": self.run_name,
				"timestamp": self.start_time,
			},
		)

	def __del__(self):
		"""Destructor to ensure logs are flushed when Logger object is destroyed"""
		try:
			if hasattr(self, "buffer") and self.buffer:
				self.flush()
		except Exception:
			# Ignore any errors during cleanup
			pass

	def record(self, event_type: str, data: Dict[str, Any]) -> None:
		"""
		Record an event - always succeeds, just adds to buffer.

		Args:
		    event_type: Type of event (e.g., "llm_call", "optimization_pass", "validation_result")
		    data: Event data dictionary
		"""
		# Serialize complex objects in data
		serialized_data = self._serialize_data(data)
		entry = {"timestamp": time.time(), "type": event_type, "data": serialized_data}
		self.buffer.append(entry)

		# Also log to console for immediate visibility
		logging.debug(f"Logger: {event_type} - {data}")

	def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Recursively serialize data, converting complex objects to dictionaries.

		Args:
		    data: Data dictionary to serialize

		Returns:
		    Serialized data dictionary
		"""
		serialized = {}
		for key, value in data.items():
			serialized[key] = self._serialize_value(value)

		# Automatically add diff_lines when diff is present
		if "diff" in serialized and "diff_lines" not in serialized:
			diff_value = serialized["diff"]
			if isinstance(diff_value, str):
				serialized["diff_lines"] = diff_value.split("\n") if diff_value else []

		return serialized

	def _serialize_value(self, value: Any) -> Any:
		"""
		Serialize a single value, handling various types.

		Args:
		    value: Value to serialize

		Returns:
		    Serialized value
		"""
		# Handle basic types
		if value is None or isinstance(value, (str, int, float, bool)):
			return value

		# Handle lists and tuples
		if isinstance(value, (list, tuple)):
			return [self._serialize_value(item) for item in value]

		# Handle dictionaries
		if isinstance(value, dict):
			result = {k: self._serialize_value(v) for k, v in value.items()}
			# Add diff_lines if diff exists and diff_lines doesn't
			if "diff" in result and "diff_lines" not in result:
				diff_value = result["diff"]
				if isinstance(diff_value, str):
					result["diff_lines"] = diff_value.split("\n") if diff_value else []
			return result

		# Handle objects with __dict__
		if hasattr(value, "__dict__"):
			obj_dict = {}
			for attr, attr_value in value.__dict__.items():
				if not attr.startswith("_"):
					obj_dict[attr] = self._serialize_value(attr_value)
			return obj_dict

		# Handle objects with accessible attributes via dir()
		if hasattr(value, "__class__"):
			obj_dict = {}
			for attr in dir(value):
				if not attr.startswith("_"):
					try:
						attr_value = getattr(value, attr)
						if not callable(attr_value):
							obj_dict[attr] = self._serialize_value(attr_value)
					except Exception:
						pass
			if obj_dict:
				return obj_dict

		# Fall back to string representation
		return str(value)

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
			"events": self.buffer,
		}

	def save_iteration_code(self, kernel_name: str, iteration_num: int, code_content: str) -> str:
		"""
		Save iteration code to a file in the trace directory.

		Args:
		    kernel_name: Name of the kernel being optimized
		    iteration_num: Iteration number
		    code_content: The code content to save

		Returns:
		    The path to the saved file
		"""
		# Get output directory from trace_path if available
		if hasattr(self, 'trace_path') and self.trace_path:
			output_dir = os.path.abspath(self.trace_path)
			os.makedirs(output_dir, exist_ok=True)
		else:
			output_dir = os.path.abspath(".")

		iteration_file = os.path.join(output_dir, f"{kernel_name}_iteration_{iteration_num}.hip")
		with open(iteration_file, "w") as f:
			f.write(code_content)
		logging.info(f"Saved iteration {iteration_num} code to {iteration_file}")

		return iteration_file

	def flush(self, output_file: Optional[str] = None) -> bool:
		"""
		Flush logs to output targets with error handling.

		Args:
		    output_file: Optional file path or directory for JSON output

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
			for event_type, count in run_summary["event_counts"].items():
				logging.info(f"  {event_type}: {count}")
			logging.info("==================")
		except Exception as e:
			logging.error(f"Console flush failed: {e}")
			success = False

		# Try file output if specified
		if output_file:
			try:
				# If output_file is a directory, create a timestamped filename
				if os.path.isdir(output_file) or not os.path.splitext(output_file)[1]:
					# Create directory if it doesn't exist
					os.makedirs(output_file, exist_ok=True)
					# Generate timestamped filename
					timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
					filename = f"intelliperf_run_{timestamp}.json"
					filepath = os.path.join(output_file, filename)
				else:
					# output_file is already a full file path
					filepath = output_file
					# Create directory if it doesn't exist
					os.makedirs(os.path.dirname(filepath), exist_ok=True)

				with open(filepath, "w") as f:
					json.dump(run_summary, f, indent=2)
				logging.info(f"Logs flushed to: {filepath}")
			except Exception as e:
				logging.error(f"File flush failed: {e}")
				success = False

		return success
