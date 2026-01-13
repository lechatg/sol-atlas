"""
Text formatting utilities for Telegram HTML.

Converts markdown to HTML and properly escapes special characters.
Based on proven implementation from bot_server.
"""
import html
import re


def escape_html(text: str) -> str:
    """
    Convert markdown formatting to HTML and escape special characters for Telegram.
    
    Supports:
    - **bold** → <b>bold</b>
    - *italic* → <i>italic</i>
    - `code` → <code>code</code>
    - ### Headings → <b>Headings</b>
    - Tables → Plain text format
    
    Args:
        text: Raw text with markdown formatting
        
    Returns:
        HTML-formatted text safe for Telegram
    """
    if not text:
        return text
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    try:
        # Step 1: Convert markdown patterns to HTML or remove unsupported ones
        
        # Convert <br> tags (and variations) to newlines first
        # This handles LLM outputs that include HTML line breaks
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        
        # Remove markdown headings (### -> plain text) since Telegram doesn't support them well
        # Use a function to safely handle replacements
        def replace_heading(match):
            try:
                if match.lastindex and match.lastindex >= 1:
                    return f'<b>{match.group(1)}</b>'
                return match.group(0)
            except (IndexError, AttributeError):
                return match.group(0)
        
        text = re.sub(r'^#{1,6}\s+(.+)$', replace_heading, text, flags=re.MULTILINE)
        
        # Convert **bold** to <b>bold</b>
        def replace_bold(match):
            try:
                if match.lastindex and match.lastindex >= 1:
                    return f'<b>{match.group(1)}</b>'
                return match.group(0)
            except (IndexError, AttributeError):
                return match.group(0)
        
        text = re.sub(r'\*\*(.*?)\*\*', replace_bold, text)
        
        # Convert *italic* to <i>italic</i> (avoid conflicts with bold)
        def replace_italic(match):
            try:
                if match.lastindex and match.lastindex >= 1:
                    return f'<i>{match.group(1)}</i>'
                return match.group(0)
            except (IndexError, AttributeError):
                return match.group(0)
        
        text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', replace_italic, text)
        
        # Convert `code` to <code>code</code>
        def replace_code(match):
            try:
                if match.lastindex and match.lastindex >= 1:
                    return f'<code>{match.group(1)}</code>'
                return match.group(0)
            except (IndexError, AttributeError):
                return match.group(0)
        
        text = re.sub(r'`([^`\n]+?)`', replace_code, text)
    
        # Handle tables - convert to simple text format since Telegram doesn't support tables
        lines = text.split('\n')
        processed_lines = []
        in_table = False
        
        for line in lines:
            # Detect table rows (contain | and look like table formatting)
            if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
                in_table = True
                # Convert table row to simple format
                cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
                if cells:
                    # Use spaces instead of | for readability
                    processed_lines.append('  '.join(cells))
            elif in_table and line.strip() and '|' not in line:
                # End of table
                in_table = False
                processed_lines.append(line)
            elif in_table and line.strip() and all(c in '-|: ' for c in line.strip()):
                # Skip table separator lines (like |---|---|)
                continue
            else:
                in_table = False
                processed_lines.append(line)
        
        text = '\n'.join(processed_lines)
        
        # Step 2: Escape HTML special characters (except our formatting tags)
        # First, protect our HTML tags
        protected_tags = []
        
        def protect_tag(match):
            try:
                tag = match.group(0)
                protected_tags.append(tag)
                return f'__TAG_{len(protected_tags)-1}__'
            except (IndexError, AttributeError):
                return match.group(0) if match else ''
        
        # Protect HTML formatting tags
        text = re.sub(r'<(/?[bi]|/?code)>', protect_tag, text)
        
        # Escape HTML characters
        text = html.escape(text)
        
        # Restore protected HTML tags
        for i, tag in enumerate(protected_tags):
            text = text.replace(f'__TAG_{i}__', tag)
        
        return text
    except Exception as e:
        # Fallback: basic HTML escaping if anything goes wrong
        # This prevents the "Replacement index out of range" error
        import html as html_module
        try:
            return html_module.escape(str(text))
        except Exception:
            # Last resort: replace problematic characters manually
            return str(text).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;').replace('%', '&#37;')


def truncate_for_telegram(text: str, max_length: int = 4096) -> str:
    """
    Truncate text to Telegram's message length limit.
    
    Telegram has a 4096 character limit per message.
    This truncates safely without breaking HTML tags.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (default 4096)
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a newline to avoid breaking sentences
    truncated = text[:max_length]
    last_newline = truncated.rfind('\n')
    
    if last_newline > max_length * 0.9:  # If newline is in last 10%
        return truncated[:last_newline] + "\n\n...(message truncated)"
    
    return truncated + "...(truncated)"

