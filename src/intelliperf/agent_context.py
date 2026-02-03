"""Context window management for the agent."""


SUMMARIZATION_PROMPT = """You are helping manage context for a GPU kernel optimization session.

Below is the conversation history so far. Summarize the KEY FACTS that must be remembered:
- Baseline performance (if established)
- Current best performance and speedup achieved
- What executable/binary is being optimized
- What optimization strategies were tried
- What worked and what didn't work
- Current bottlenecks identified

Keep it concise (5-10 bullet points max). Focus on facts needed to continue optimization.

Conversation to summarize:
{conversation}

Provide ONLY the bullet-point summary, no extra commentary."""


def _fallback_summary(conversation_text, error=None):
    """Fallback summary when LLM summarization fails."""
    prefix = "(Fallback summary: LLM summarization failed)" if error else "(Fallback summary: empty response)"
    summary_lines = [prefix]

    # Keep only the last 10 lines to stay concise
    tail = conversation_text[-10:]
    summary_lines.extend(f"- {line[:200]}" for line in tail)
    return "\n".join(summary_lines)


def summarize_conversation(llm, messages):
    """
    Use LLM to summarize conversation history.

    Args:
        llm: LLM instance
        messages: Full conversation history

    Returns:
        Summary string
    """
    # Skip system message, get the conversation
    conversation_text = []
    for msg in messages[1:]:  # Skip system message
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Truncate very long tool results
        if len(content) > 500:
            content = content[:500] + "\n...[truncated]"

        conversation_text.append(f"{role.upper()}: {content}")

    conversation_str = "\n\n".join(conversation_text)

    # Ask LLM to summarize
    summary_messages = [{
        "role": "user",
        "content": SUMMARIZATION_PROMPT.format(conversation=conversation_str)
    }]

    try:
        response = llm.invoke(summary_messages)
    except Exception as e:
        return _fallback_summary(conversation_text, error=e)

    if hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)

    if not content or not str(content).strip():
        return _fallback_summary(conversation_text, error=None)

    return content


def manage_context_window(llm, messages, system_message, initial_message, context_window_size):
    """
    Manage context window using LLM-based summarization.

    Keep:
    - System message (always)
    - Initial goal (always)
    - LLM-generated summary of previous context
    - Last N message pairs

    Args:
        llm: LLM instance for summarization
        messages: Full conversation history
        system_message: System prompt
        initial_message: Initial goal/task
        context_window_size: Number of recent message pairs to keep

    Returns:
        Pruned message list
    """
    if len(messages) <= context_window_size + 2:  # +2 for system and initial
        return messages

    # Get LLM summary of what's important
    summary = summarize_conversation(llm, messages)

    # Build new context
    new_messages = [system_message]

    # Add summary
    new_messages.append({
        "role": "user",
        "content": f"**Context Summary (from previous iterations):**\n\n{summary}"
    })

    # Add initial goal
    new_messages.append(initial_message)

    # Keep last N messages (excluding system and initial)
    recent_messages = messages[2:]  # Skip system and initial
    keep_count = context_window_size * 2  # *2 for assistant+user pairs

    if len(recent_messages) > keep_count:
        new_messages.extend(recent_messages[-keep_count:])
    else:
        new_messages.extend(recent_messages)

    return new_messages
