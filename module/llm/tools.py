"""
Tools for the LLM to interact with the assistant system.
"""

import json
import datetime
import platform
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tools that can be used by the LLM"""

    def __init__(self):
        """Initialize tool registry"""
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, function: Callable):
        """
        Register a tool

        Args:
            name: Tool name
            description: Tool description
            function: Tool function
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "function": function,
        }
        logger.info(f"Registered tool: {name}")

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get tool descriptions for LLM

        Returns:
            List of tool descriptions
        """
        return [
            {
                "type": "function",
                "function": {"name": name, "description": tool["description"]},
            }
            for name, tool in self.tools.items()
        ]

    def execute_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a tool

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")

        tool = self.tools[name]
        try:
            if arguments:
                return tool["function"](**arguments)
            else:
                return tool["function"]()
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return {"error": str(e)}


# Create global tool registry
tool_registry = ToolRegistry()


# System information tools
def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_system_resources() -> Dict[str, Any]:
    """Get system resource usage"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }


# Register basic tools
tool_registry.register(
    "get_system_info",
    "Get system information including platform, processor, and current time",
    get_system_info,
)

tool_registry.register(
    "get_system_resources",
    "Get current system resource usage including CPU, memory, and disk",
    get_system_resources,
)
