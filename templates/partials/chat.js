/**
 * Chat Module
 * Universal chat modal for Coachd app
 * 
 * Usage:
 *   1. Include partials/chat.html in your page
 *   2. Include partials/chat_styles.css in your styles
 *   3. Include this script
 *   4. Optionally set Chat.getAgencyCode = () => yourAgencyCode
 *   5. Call Chat.open(), Chat.close(), Chat.send(), etc.
 */
const Chat = (function() {
    let messages = [];
    let isWaiting = false;
    
    // Override this in your page to provide agency context
    let getAgencyCode = () => null;
    
    function init() {
        const input = document.getElementById('chatInput');
        if (!input) return;
        
        // Auto-resize textarea
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 100) + 'px';
        });
        
        // Enter to send
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        });
    }
    
    function open() {
        document.getElementById('chatOverlay').classList.add('open');
        document.getElementById('chatInput').focus();
    }
    
    function close() {
        document.getElementById('chatOverlay').classList.remove('open');
    }
    
    function openWith(query) {
        open();
        if (query) {
            document.getElementById('chatInput').value = query;
            setTimeout(() => send(), 100);
        }
    }
    
    function useSuggestion(text) {
        document.getElementById('chatInput').value = text;
        send();
    }
    
    async function send() {
        const input = document.getElementById('chatInput');
        const text = input.value.trim();
        
        if (!text || isWaiting) return;
        
        // Hide empty state
        document.getElementById('chatEmpty').style.display = 'none';
        
        const messagesEl = document.getElementById('chatMessages');
        
        // Add user message
        addMessage('user', text);
        input.value = '';
        input.style.height = 'auto';
        
        // Set waiting state
        isWaiting = true;
        document.getElementById('chatSend').disabled = true;
        
        // Show typing indicator
        const typingEl = document.createElement('div');
        typingEl.className = 'chat-message ai';
        typingEl.innerHTML = '<div class="msg-label">Coachd</div><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
        messagesEl.appendChild(typingEl);
        scrollToBottom();
        
        try {
            const res = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    agency: getAgencyCode(),
                    history: messages.slice(-10)
                })
            });
            
            typingEl.remove();
            
            if (!res.ok) throw new Error();
            
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let fullText = '';
            
            // Create AI message element
            const aiMsgEl = document.createElement('div');
            aiMsgEl.className = 'chat-message ai';
            aiMsgEl.innerHTML = '<div class="msg-label">Coachd</div><div class="msg-bubble"></div>';
            messagesEl.appendChild(aiMsgEl);
            const bubbleEl = aiMsgEl.querySelector('.msg-bubble');
            
            // Stream response
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n\n');
                buffer = parts.pop() || '';
                
                for (const part of parts) {
                    for (const line of part.split('\n')) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;
                            
                            try {
                                const parsed = JSON.parse(data);
                                const chunk = parsed.text || parsed.content;
                                if (chunk) {
                                    fullText += chunk;
                                    bubbleEl.innerHTML = formatMessage(fullText);
                                    scrollToBottom();
                                }
                            } catch (e) {
                                if (data.trim() && data !== '[DONE]') {
                                    fullText += data;
                                    bubbleEl.innerHTML = formatMessage(fullText);
                                    scrollToBottom();
                                }
                            }
                        }
                    }
                }
            }
            
            bubbleEl.innerHTML = formatMessage(fullText);
            messages.push({ role: 'assistant', content: fullText });
            
        } catch (e) {
            typingEl.remove();
            addMessage('ai', 'Sorry, something went wrong. Please try again.');
        }
        
        isWaiting = false;
        document.getElementById('chatSend').disabled = false;
    }
    
    function addMessage(role, content) {
        const messagesEl = document.getElementById('chatMessages');
        const msgEl = document.createElement('div');
        msgEl.className = `chat-message ${role}`;
        
        if (role === 'user') {
            msgEl.innerHTML = `<div class="msg-bubble">${escapeHtml(content)}</div>`;
        } else {
            msgEl.innerHTML = `<div class="msg-label">Coachd</div><div class="msg-bubble">${formatMessage(content)}</div>`;
        }
        
        messagesEl.appendChild(msgEl);
        messages.push({ role: role === 'ai' ? 'assistant' : role, content });
        scrollToBottom();
    }
    
    function formatMessage(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
    }
    
    function escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
    
    function scrollToBottom() {
        const body = document.getElementById('chatBody');
        body.scrollTop = body.scrollHeight;
    }
    
    function clear() {
        messages = [];
        document.getElementById('chatMessages').innerHTML = '';
        document.getElementById('chatEmpty').style.display = 'flex';
    }
    
    function getHistory() {
        return messages;
    }
    
    function setAgencyCodeGetter(fn) {
        getAgencyCode = fn;
    }
    
    // Auto-init when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // Public API
    return {
        open,
        close,
        openWith,
        send,
        useSuggestion,
        clear,
        getHistory,
        setAgencyCodeGetter,
        init
    };
})();
