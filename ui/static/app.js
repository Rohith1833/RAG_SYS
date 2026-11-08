/**
 * ULTRON AI Assistant & Laboratory — Master Frontend Controller
 * Complete Integration:
 * 1. Dual-Mode Assistant (General Chat + Document-Grounded Semantic RAG)
 * 2. SQLite Persistent Conversations & Message History
 * 3. Document Ingestion (TXT, MD, PDF, DOCX) & Chunk Registry
 * 4. Hierarchical Complete-Document Summarizer Modal
 * 5. Markdown & Code Block Highlighting with Copy Actions
 * 6. Preserved Ultron v0.2 TinyStories PyTorch Laboratory
 */

document.addEventListener('DOMContentLoaded', () => {

  // =========================================================================
  // 1. Core State
  // =========================================================================
  let currentMode = 'rag'; // 'rag' or 'general'
  let currentConvId = null;
  let isGenerating = false;
  let lastUserMessage = '';
  let activeSummarizeDocId = null;
  let selectedSummarizeStyle = 'detailed';

  // =========================================================================
  // 2. Navigation & View Switcher
  // =========================================================================
  const tabRagBtn = document.getElementById('tabRagBtn');
  const tabLabBtn = document.getElementById('tabLabBtn');
  const ragView = document.getElementById('ragView');
  const labView = document.getElementById('labView');

  function switchMainTab(target) {
    if (target === 'ragView') {
      tabRagBtn.classList.add('active');
      tabLabBtn.classList.remove('active');
      ragView.classList.remove('hidden');
      ragView.classList.add('active-view');
      labView.classList.add('hidden');
      labView.classList.remove('active-view');
    } else {
      tabLabBtn.classList.add('active');
      tabRagBtn.classList.remove('active');
      labView.classList.remove('hidden');
      labView.classList.add('active-view');
      ragView.classList.add('hidden');
      ragView.classList.remove('active-view');
    }
  }

  if (tabRagBtn && tabLabBtn) {
    tabRagBtn.addEventListener('click', () => switchMainTab('ragView'));
    tabLabBtn.addEventListener('click', () => switchMainTab('labView'));
  }

  // =========================================================================
  // 3. Global Notification Banner
  // =========================================================================
  const errorBanner = document.getElementById('errorBanner');
  const errorMessage = document.getElementById('errorMessage');
  const dismissError = document.getElementById('dismissError');

  function showError(msg) {
    if (errorMessage && errorBanner) {
      errorMessage.textContent = msg;
      errorBanner.classList.remove('hidden');
    }
  }

  function hideError() {
    if (errorBanner) {
      errorBanner.classList.add('hidden');
    }
  }

  if (dismissError) {
    dismissError.addEventListener('click', hideError);
  }

  // =========================================================================
  // 4. System Telemetry & Health Monitoring
  // =========================================================================
  const statusBadge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
  const telemetryModel = document.getElementById('telemetryModel');
  const telemetryOllama = document.getElementById('telemetryOllama');
  const telemetryEmbedder = document.getElementById('telemetryEmbedder');
  const telemetryDim = document.getElementById('telemetryDim');
  const telemetryDevice = document.getElementById('telemetryDevice');
  const telemetryChunks = document.getElementById('telemetryChunks');
  const docCountBadge = document.getElementById('docCountBadge');
  const inputModelPill = document.getElementById('inputModelPill');

  async function fetchHealthTelemetry() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error(`Health HTTP ${res.status}`);
      const data = await res.json();

      const deviceStr = (data.device || 'CUDA').toUpperCase();
      const modelStr = data.model || 'llama3:latest';

      if (data.ollama) {
        statusBadge.className = 'status-pill clay-pill online';
        statusText.textContent = `Online • ${deviceStr} (${modelStr})`;
        hideError();
      } else {
        statusBadge.className = 'status-pill clay-pill offline-warning';
        statusText.textContent = 'Ollama Offline • Ingestion & Retrieval Ready';
      }

      if (telemetryModel) telemetryModel.textContent = modelStr;
      if (inputModelPill) inputModelPill.textContent = data.ollama ? `Ollama (${modelStr})` : 'Ollama Offline';
      if (telemetryOllama) {
        telemetryOllama.textContent = data.ollama ? 'Connected (Local)' : 'Offline (Generation Unavailable)';
        telemetryOllama.style.color = data.ollama ? 'var(--status-green)' : '#D97706';
      }
      if (telemetryEmbedder) telemetryEmbedder.textContent = data.embedding_model || 'all-MiniLM-L6-v2';
      if (telemetryDim) telemetryDim.textContent = data.embedding_dimension || '384';
      if (telemetryDevice) telemetryDevice.textContent = deviceStr;
      if (telemetryChunks) telemetryChunks.textContent = `${data.chunks || 0} chunk(s) in memory`;
      if (docCountBadge) docCountBadge.textContent = `${data.documents || 0} Documents`;

    } catch (err) {
      statusBadge.className = 'status-pill clay-pill error';
      statusText.textContent = 'Backend Error';
    }
  }

  fetchHealthTelemetry();
  setInterval(fetchHealthTelemetry, 20000);

  // =========================================================================
  // 5. Mode Switching (Document RAG vs General Chat)
  // =========================================================================
  const modeOptRag = document.getElementById('modeOptRag');
  const modeOptGeneral = document.getElementById('modeOptGeneral');
  const activeModeBadge = document.getElementById('activeModeBadge');
  const inputModePillTag = document.getElementById('inputModePillTag');
  const ragPromptInput = document.getElementById('ragPromptInput');

  function setMode(mode) {
    currentMode = mode;
    if (mode === 'rag') {
      modeOptRag.classList.add('active');
      modeOptGeneral.classList.remove('active');
      if (activeModeBadge) {
        activeModeBadge.textContent = 'DOCUMENT RAG';
        activeModeBadge.style.color = 'var(--primary-coral)';
      }
      if (inputModePillTag) {
        inputModePillTag.className = 'active-mode-pill-tag';
        inputModePillTag.innerHTML = '<span class="pill-dot"></span> Mode: Document RAG';
      }
      if (ragPromptInput) {
        ragPromptInput.placeholder = 'Ask a question about your uploaded documents... Press Enter to send';
      }
    } else {
      modeOptGeneral.classList.add('active');
      modeOptRag.classList.remove('active');
      if (activeModeBadge) {
        activeModeBadge.textContent = 'GENERAL CHAT';
        activeModeBadge.style.color = '#2563EB';
      }
      if (inputModePillTag) {
        inputModePillTag.className = 'active-mode-pill-tag general';
        inputModePillTag.innerHTML = '<span class="pill-dot"></span> Mode: General Chat';
      }
      if (ragPromptInput) {
        ragPromptInput.placeholder = 'Ask anything (coding, reasoning, concepts)... Press Enter to send';
      }
    }
  }

  if (modeOptRag && modeOptGeneral) {
    modeOptRag.addEventListener('click', () => setMode('rag'));
    modeOptGeneral.addEventListener('click', () => setMode('general'));
  }

  // =========================================================================
  // 6. Conversation Management (SQLite Persistent)
  // =========================================================================
  const newChatBtn = document.getElementById('newChatBtn');
  const convListContainer = document.getElementById('convListContainer');
  const convCountBadge = document.getElementById('convCountBadge');
  const activeConvTitle = document.getElementById('activeConvTitle');

  async function fetchConversations() {
    try {
      const res = await fetch('/api/conversations');
      if (!res.ok) return;
      const data = await res.json();
      renderConversations(data.conversations || []);
    } catch (e) {
      console.error('Failed to load conversations:', e);
    }
  }

  function renderConversations(convs) {
    if (!convListContainer) return;
    if (convCountBadge) convCountBadge.textContent = convs.length;

    if (!convs || convs.length === 0) {
      convListContainer.innerHTML = `
        <div id="emptyConvState" class="empty-conv-state">
          <span>💬</span>
          <p>No conversations yet.</p>
        </div>
      `;
      return;
    }

    convListContainer.innerHTML = '';
    convs.forEach(conv => {
      const item = document.createElement('div');
      item.className = `conv-item ${conv.id === currentConvId ? 'active' : ''}`;
      item.setAttribute('data-id', conv.id);

      const modeBadge = conv.mode === 'general' ? 'general' : 'rag';
      const modeLabel = conv.mode === 'general' ? 'Chat' : 'RAG';

      item.innerHTML = `
        <div class="conv-item-info">
          <span class="conv-item-badge ${modeBadge}">${modeLabel}</span>
          <span class="conv-item-title" title="${escapeHtml(conv.title)}">${escapeHtml(conv.title)}</span>
        </div>
        <button type="button" class="conv-del-btn" data-id="${conv.id}" title="Delete conversation">✕</button>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('conv-del-btn')) return;
        selectConversation(conv.id);
      });

      const delBtn = item.querySelector('.conv-del-btn');
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm(`Delete conversation "${conv.title}"?`)) {
          await deleteConversation(conv.id);
        }
      });

      convListContainer.appendChild(item);
    });
  }

  async function selectConversation(convId) {
    currentConvId = convId;
    fetchConversations();

    try {
      const res = await fetch(`/api/conversations/${convId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (activeConvTitle) activeConvTitle.textContent = data.title || 'Conversation';
      setMode(data.mode || 'rag');

      // Clear transcript and render message history
      clearTranscriptView();
      const messages = data.messages || [];
      if (messages.length === 0) {
        showWelcomeState();
      } else {
        hideWelcomeState();
        messages.forEach(msg => {
          if (msg.role === 'user') {
            appendUserMessage(msg.content);
          } else {
            appendAssistantMessage({
              answer: msg.content,
              sources: msg.sources || [],
              mode: data.mode
            });
          }
        });
      }
    } catch (err) {
      showError(`Failed to load conversation: ${err.message}`);
    }
  }

  async function createNewConversation() {
    try {
      const res = await fetch('/api/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Conversation', mode: currentMode })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const conv = await res.json();
      currentConvId = conv.id;
      if (activeConvTitle) activeConvTitle.textContent = 'New Conversation';
      clearTranscriptView();
      showWelcomeState();
      fetchConversations();
    } catch (err) {
      showError(`Failed to create conversation: ${err.message}`);
    }
  }

  async function deleteConversation(convId) {
    try {
      const res = await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (currentConvId === convId) {
        currentConvId = null;
        if (activeConvTitle) activeConvTitle.textContent = 'Ultron Assistant';
        clearTranscriptView();
        showWelcomeState();
      }
      fetchConversations();
    } catch (err) {
      showError(`Failed to delete conversation: ${err.message}`);
    }
  }

  if (newChatBtn) {
    newChatBtn.addEventListener('click', createNewConversation);
  }

  // =========================================================================
  // 7. Chat Transcript & Markdown Renderer
  // =========================================================================
  const ragTranscript = document.getElementById('ragTranscript');
  let ragWelcomeState = document.getElementById('ragWelcomeState');
  const ragSendBtn = document.getElementById('ragSendBtn');
  const ragBtnSpinner = document.getElementById('ragBtnSpinner');
  const ragBtnLabel = document.getElementById('ragBtnLabel');
  const clearRagChatBtn = document.getElementById('clearRagChatBtn');

  function scrollToBottom() {
    if (ragTranscript) {
      ragTranscript.scrollTop = ragTranscript.scrollHeight;
    }
  }

  function clearTranscriptView() {
    if (ragTranscript) {
      ragTranscript.innerHTML = '';
    }
  }

  function showWelcomeState() {
    if (!ragWelcomeState) {
      ragWelcomeState = document.createElement('div');
      ragWelcomeState.id = 'ragWelcomeState';
      ragWelcomeState.className = 'welcome-card clay-card-subtle';
      ragWelcomeState.innerHTML = `
        <div class="welcome-icon">✦</div>
        <h3>Ultron Local AI Assistant</h3>
        <p>
          Ask questions in <strong>Document RAG mode</strong> over indexed documents, or switch to 
          <strong>General Chat mode</strong> for unconstrained dialogue with local <strong>llama3:latest</strong>.
        </p>
        <div class="starter-suggestions">
          <span class="suggestion-label">Suggested prompts:</span>
          <div class="suggestion-pills">
            <button type="button" class="clay-chip rag-chip" data-prompt="What is Ultron and how is it architected?">"What is Ultron?"</button>
            <button type="button" class="clay-chip rag-chip" data-prompt="Explain how self-attention works in a Transformer.">"Explain self-attention"</button>
            <button type="button" class="clay-chip rag-chip" data-prompt="Write a Python script to compute cosine similarity using NumPy.">"Python cosine similarity"</button>
          </div>
        </div>
      `;
      setupSuggestionChips(ragWelcomeState);
    }
    if (ragTranscript && !document.getElementById('ragWelcomeState')) {
      ragTranscript.appendChild(ragWelcomeState);
    }
  }

  function hideWelcomeState() {
    const el = document.getElementById('ragWelcomeState');
    if (el) el.remove();
  }

  function setupSuggestionChips(container) {
    const chips = container.querySelectorAll('.rag-chip');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        const prompt = chip.getAttribute('data-prompt');
        if (ragPromptInput) {
          ragPromptInput.value = prompt;
          handleSendQuery();
        }
      });
    });
  }

  setupSuggestionChips(document);

  function appendUserMessage(text) {
    hideWelcomeState();
    const row = document.createElement('div');
    row.className = 'message-bubble-row user-row';
    row.innerHTML = `
      <div class="message-meta"><span>You</span></div>
      <div class="user-bubble">${escapeHtml(text)}</div>
    `;
    ragTranscript.appendChild(row);
    scrollToBottom();
  }

  function createGeneratingBubble() {
    const row = document.createElement('div');
    row.id = 'ragGeneratingIndicator';
    row.className = 'message-bubble-row ultron-row';
    const textMsg = currentMode === 'rag' 
      ? 'Retrieving relevant passages & synthesizing answer with Llama 3...'
      : 'Thinking with llama3:latest...';

    row.innerHTML = `
      <div class="ultron-bubble generating-bubble">
        <div class="pulse-spinner"></div>
        <span class="generating-text">${textMsg}</span>
      </div>
    `;
    ragTranscript.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendAssistantMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-bubble-row ultron-row';

    const bubble = document.createElement('div');
    bubble.className = 'ultron-bubble';

    // Header
    const head = document.createElement('div');
    head.className = 'bubble-head';

    const isOffline = (data.generation_available === false || data.ollama_available === false);
    const badge = document.createElement('div');
    badge.className = 'agent-badge';

    if (isOffline) {
      badge.style.color = '#B45309';
      badge.innerHTML = `
        <span class="agent-avatar-mini" style="background: #D97706;">✦</span>
        <span>Ultron (Ollama Offline)</span>
      `;
    } else {
      const modeColor = data.mode === 'general' ? '#2563EB' : 'var(--primary-coral)';
      badge.innerHTML = `
        <span class="agent-avatar-mini" style="background: ${modeColor};">✦</span>
        <span>Ultron ${data.mode === 'general' ? 'General Chat' : 'Grounded RAG'}</span>
      `;
    }

    const telemetry = document.createElement('span');
    telemetry.className = 'telemetry-chip';
    const latStr = data.retrieval_ms ? `Retrieved in ${data.retrieval_ms}ms` : 'Local Inference';
    telemetry.textContent = latStr;

    head.appendChild(badge);
    head.appendChild(telemetry);
    bubble.appendChild(head);

    // Markdown Answer Content
    const content = document.createElement('div');
    content.className = 'markdown-body';
    content.innerHTML = renderMarkdown(data.answer || '');
    bubble.appendChild(content);

    // Render Sources if present (RAG mode)
    if (data.sources && data.sources.length > 0) {
      const sourcesCard = document.createElement('div');
      sourcesCard.className = 'sources-card';

      const sourcesHead = document.createElement('div');
      sourcesHead.className = 'sources-head';
      sourcesHead.innerHTML = `
        <span class="sources-title">
          <span>📚</span> Grounded Sources Used (${data.sources.length})
        </span>
      `;
      sourcesCard.appendChild(sourcesHead);

      const sourcesList = document.createElement('div');
      sourcesList.className = 'sources-list';

      data.sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';

        const pageLabel = src.page ? `Page ${src.page} • ` : '';
        const scoreLabel = src.score !== undefined ? `Similarity: ${Number(src.score).toFixed(4)}` : '';

        item.innerHTML = `
          <div class="source-item-header">
            <span class="source-filename">
              <span>📄</span> ${escapeHtml(src.filename)} (${pageLabel}Chunk ${src.chunk_id})
            </span>
            <span class="source-score-pill">${scoreLabel}</span>
          </div>
          <div class="source-preview">"${escapeHtml(src.preview)}"</div>
        `;
        sourcesList.appendChild(item);
      });

      sourcesCard.appendChild(sourcesList);
      bubble.appendChild(sourcesCard);
    }

    // Footer Actions: Copy Response & Regenerate
    const footer = document.createElement('div');
    footer.className = 'bubble-footer-actions';

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'action-pill-btn';
    copyBtn.textContent = 'Copy response';
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(data.answer || '');
        copyBtn.textContent = '✓ Copied!';
        setTimeout(() => copyBtn.textContent = 'Copy response', 1500);
      } catch (e) {
        copyBtn.textContent = 'Copy failed';
      }
    });

    const regenBtn = document.createElement('button');
    regenBtn.type = 'button';
    regenBtn.className = 'action-pill-btn';
    regenBtn.textContent = '↻ Regenerate';
    regenBtn.addEventListener('click', () => {
      if (lastUserMessage && !isGenerating) {
        ragPromptInput.value = lastUserMessage;
        handleSendQuery();
      }
    });

    footer.appendChild(copyBtn);
    footer.appendChild(regenBtn);
    bubble.appendChild(footer);

    row.appendChild(bubble);
    ragTranscript.appendChild(row);
    scrollToBottom();
  }

  // =========================================================================
  // 8. Markdown & Code Block Formatter
  // =========================================================================
  function renderMarkdown(rawText) {
    if (!rawText) return '';

    // Escape HTML special characters first
    let escaped = escapeHtml(rawText);

    // 1. Code blocks: ```lang ... ```
    escaped = escaped.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const languageTag = lang ? lang.toUpperCase() : 'CODE';
      return `
        <pre><div class="code-header-bar"><span>${languageTag}</span><button type="button" class="copy-code-button" onclick="navigator.clipboard.writeText(this.closest('pre').querySelector('code').innerText); this.textContent='✓ Copied'; setTimeout(() => this.textContent='Copy', 1500);">Copy</button></div><code>${code.trim()}</code></pre>
      `;
    });

    // 2. Inline code: `code`
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 3. Bold: **text**
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // 4. Italic: *text*
    escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // 5. Unordered lists: lines starting with "- " or "* "
    escaped = escaped.replace(/(?:^|\n)[-\*]\s+(.+)/g, '\n<li>$1</li>');
    escaped = escaped.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');

    // 6. Numbered lists: lines starting with "1. "
    escaped = escaped.replace(/(?:^|\n)\d+\.\s+(.+)/g, '\n<li>$1</li>');

    // 7. Paragraphs: split by double newline
    const paragraphs = escaped.split(/\n{2,}/);
    const htmlParas = paragraphs.map(p => {
      const trimmed = p.trim();
      if (!trimmed) return '';
      if (trimmed.startsWith('<pre') || trimmed.startsWith('<ul') || trimmed.startsWith('<ol')) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    });

    return htmlParas.join('');
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // =========================================================================
  // 9. Send Chat Query Handler (Real-Time Token Streaming with Stop Support)
  // =========================================================================
  let activeAbortController = null;

  function createStreamingBubble(meta) {
    const row = document.createElement('div');
    row.className = 'message-bubble-row ultron-row';

    const bubble = document.createElement('div');
    bubble.className = 'ultron-bubble';

    // Header
    const head = document.createElement('div');
    head.className = 'bubble-head';

    const mode = meta.mode || currentMode;
    const modeColor = mode === 'general' ? '#2563EB' : 'var(--primary-coral)';
    const badge = document.createElement('div');
    badge.className = 'agent-badge';
    badge.innerHTML = `
      <span class="agent-avatar-mini" style="background: ${modeColor};">✦</span>
      <span>Ultron ${mode === 'general' ? 'General Chat' : 'Grounded RAG'}</span>
    `;

    const telemetry = document.createElement('span');
    telemetry.className = 'telemetry-chip';
    telemetry.textContent = meta.retrieval_ms ? `Retrieved in ${meta.retrieval_ms}ms` : 'Generating...';

    head.appendChild(badge);
    head.appendChild(telemetry);
    bubble.appendChild(head);

    // Markdown Answer Content
    const content = document.createElement('div');
    content.className = 'markdown-body';
    content.innerHTML = '<span class="streaming-cursor">▊</span>';
    bubble.appendChild(content);

    // Render Sources if present (RAG mode)
    if (meta.sources && meta.sources.length > 0) {
      const sourcesCard = document.createElement('div');
      sourcesCard.className = 'sources-card';
      sourcesCard.innerHTML = `
        <div class="sources-head">
          <span class="sources-title">
            <span>📚</span> Grounded Sources Used (${meta.sources.length})
          </span>
        </div>
      `;

      const sourcesList = document.createElement('div');
      sourcesList.className = 'sources-list';

      meta.sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';
        const pageLabel = src.page ? `Page ${src.page} • ` : '';
        const scoreLabel = src.score !== undefined ? `Similarity: ${Number(src.score).toFixed(4)}` : '';

        item.innerHTML = `
          <div class="source-item-header">
            <span class="source-filename">
              <span>📄</span> ${escapeHtml(src.filename)} (${pageLabel}Chunk ${src.chunk_id})
            </span>
            <span class="source-score-pill">${scoreLabel}</span>
          </div>
          <div class="source-preview">"${escapeHtml(src.preview)}"</div>
        `;
        sourcesList.appendChild(item);
      });

      sourcesCard.appendChild(sourcesList);
      bubble.appendChild(sourcesCard);
    }

    // Footer container
    const footer = document.createElement('div');
    footer.className = 'bubble-footer-actions';
    bubble.appendChild(footer);

    row.appendChild(bubble);
    ragTranscript.appendChild(row);
    scrollToBottom();

    return {
      row,
      bubble,
      content,
      telemetry,
      footer,
      updateText(text) {
        content.innerHTML = renderMarkdown(text) + '<span class="streaming-cursor">▊</span>';
      },
      finalize(doneEvent, text) {
        content.innerHTML = renderMarkdown(text);
        const totalMs = doneEvent.total_ms || 0;
        telemetry.textContent = `${totalMs}ms • Local Inference`;

        footer.innerHTML = '';
        const copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.className = 'action-pill-btn';
        copyBtn.textContent = 'Copy response';
        copyBtn.addEventListener('click', async () => {
          try {
            await navigator.clipboard.writeText(text);
            copyBtn.textContent = '✓ Copied!';
            setTimeout(() => copyBtn.textContent = 'Copy response', 1500);
          } catch (e) {
            copyBtn.textContent = 'Copy failed';
          }
        });

        const regenBtn = document.createElement('button');
        regenBtn.type = 'button';
        regenBtn.className = 'action-pill-btn';
        regenBtn.textContent = '↻ Regenerate';
        regenBtn.addEventListener('click', () => {
          if (lastUserMessage && !isGenerating) {
            ragPromptInput.value = lastUserMessage;
            handleSendQuery();
          }
        });

        footer.appendChild(copyBtn);
        footer.appendChild(regenBtn);
      }
    };
  }

  async function handleSendQuery() {
    if (isGenerating) {
      if (activeAbortController) {
        activeAbortController.abort();
      }
      return;
    }

    const query = ragPromptInput.value.trim();
    if (!query) return;

    lastUserMessage = query;
    isGenerating = true;
    activeAbortController = new AbortController();

    // Set button to Stop state
    ragSendBtn.classList.add('stop-state');
    ragBtnLabel.textContent = '■ Stop';
    ragBtnSpinner.classList.add('hidden');
    ragBtnLabel.classList.remove('hidden');

    appendUserMessage(query);
    ragPromptInput.value = '';
    const generatingBubble = createGeneratingBubble();

    let streamState = null;
    let fullText = '';

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: activeAbortController.signal,
        body: JSON.stringify({
          message: query,
          mode: currentMode,
          conversation_id: currentConvId,
          top_k: 3
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr) continue;

          try {
            const ev = JSON.parse(jsonStr);

            if (ev.conversation_id && ev.conversation_id !== currentConvId) {
              currentConvId = ev.conversation_id;
              fetchConversations();
            }

            if (ev.type === 'meta') {
              if (generatingBubble && generatingBubble.parentNode) {
                generatingBubble.remove();
              }
              streamState = createStreamingBubble(ev);
            } else if (ev.type === 'token') {
              if (!streamState) {
                if (generatingBubble && generatingBubble.parentNode) {
                  generatingBubble.remove();
                }
                streamState = createStreamingBubble({ mode: currentMode });
              }
              fullText += ev.token;
              streamState.updateText(fullText);
              scrollToBottom();
            } else if (ev.type === 'done') {
              if (streamState) {
                streamState.finalize(ev, fullText);
              }
              fetchConversations();
            } else if (ev.type === 'error') {
              showError(ev.error);
            }
          } catch (pe) {
            console.error('SSE parse error:', pe);
          }
        }
      }

      if (streamState) {
        streamState.finalize({ total_ms: 0 }, fullText);
      }

    } catch (err) {
      if (generatingBubble && generatingBubble.parentNode) {
        generatingBubble.remove();
      }

      if (err.name === 'AbortError') {
        if (streamState) {
          streamState.finalize({ total_ms: 0 }, fullText + ' *(generation stopped by user)*');
        }
      } else {
        showError(`Query failed: ${err.message}`);
        appendAssistantMessage({
          answer: `Encountered an error: ${err.message}`,
          mode: currentMode,
          generation_available: false
        });
      }
    } finally {
      isGenerating = false;
      activeAbortController = null;
      ragSendBtn.classList.remove('stop-state');
      ragBtnLabel.textContent = 'Send';
      ragSendBtn.disabled = false;
      ragBtnSpinner.classList.add('hidden');
      ragBtnLabel.classList.remove('hidden');
      scrollToBottom();
    }
  }

  if (ragSendBtn) {
    ragSendBtn.addEventListener('click', handleSendQuery);
  }

  if (ragPromptInput) {
    ragPromptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendQuery();
      }
    });
  }

  if (clearRagChatBtn) {
    clearRagChatBtn.addEventListener('click', () => {
      clearTranscriptView();
      showWelcomeState();
    });
  }

  // =========================================================================
  // 10. Document Ingestion & Management
  // =========================================================================
  const uploadForm = document.getElementById('uploadForm');
  const fileInput = document.getElementById('fileInput');
  const dropzoneContent = document.getElementById('dropzoneContent');
  const uploadSubmitBtn = document.getElementById('uploadSubmitBtn');
  const uploadStatusBox = document.getElementById('uploadStatusBox');
  const uploadStatusText = document.getElementById('uploadStatusText');
  const uploadProgressBar = document.getElementById('uploadProgressBar');
  const docListContainer = document.getElementById('docListContainer');
  const clearKbBtn = document.getElementById('clearKbBtn');

  let selectedFile = null;

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        setFileToUpload(e.target.files[0]);
      }
    });
  }

  if (uploadForm) {
    ['dragenter', 'dragover'].forEach(name => {
      uploadForm.addEventListener(name, (e) => {
        e.preventDefault();
        uploadForm.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      uploadForm.addEventListener(name, (e) => {
        e.preventDefault();
        uploadForm.classList.remove('drag-over');
      });
    });

    uploadForm.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        setFileToUpload(files[0]);
      }
    });

    uploadForm.addEventListener('click', () => {
      if (fileInput) fileInput.click();
    });
  }

  function setFileToUpload(file) {
    const ext = file.name.slice((file.name.lastIndexOf(".") - 1 >>> 0) + 2).toLowerCase();
    if (!['txt', 'md', 'pdf', 'docx'].includes(ext)) {
      showError(`Unsupported file format .${ext}. Please upload .txt, .md, .pdf, or .docx.`);
      return;
    }
    selectedFile = file;
    dropzoneContent.innerHTML = `
      <span class="upload-icon">📄</span>
      <span class="dropzone-text" style="color: var(--primary-coral); font-weight: 700;">${escapeHtml(file.name)}</span>
      <span class="supported-formats">${(file.size / 1024).toFixed(1)} KB • Ready to index</span>
    `;
    uploadSubmitBtn.disabled = false;
  }

  if (uploadSubmitBtn) {
    uploadSubmitBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      uploadSubmitBtn.disabled = true;
      uploadStatusBox.classList.remove('hidden');
      uploadProgressBar.style.width = '25%';
      uploadStatusText.textContent = 'Uploading file to Ultron server...';

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const timer1 = setTimeout(() => {
          uploadProgressBar.style.width = '55%';
          uploadStatusText.textContent = 'Parsing text & generating chunks...';
        }, 300);

        const timer2 = setTimeout(() => {
          uploadProgressBar.style.width = '85%';
          uploadStatusText.textContent = 'Generating 384-d embeddings (MiniLM)...';
        }, 700);

        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });

        clearTimeout(timer1);
        clearTimeout(timer2);

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP ${res.status}`);
        }

        const result = await res.json();
        uploadProgressBar.style.width = '100%';
        uploadStatusText.textContent = result.status === 'already_indexed' 
          ? 'Identical document already indexed.' 
          : `Indexed ${result.chunks_indexed} chunks successfully!`;

        setTimeout(() => {
          uploadStatusBox.classList.add('hidden');
          selectedFile = null;
          fileInput.value = '';
          dropzoneContent.innerHTML = `
            <span class="upload-icon">📁</span>
            <span class="dropzone-text">Click or drag file to upload</span>
            <span class="supported-formats">Supported: <strong>.TXT</strong> • <strong>.MD</strong> • <strong>.PDF</strong> • <strong>.DOCX</strong></span>
          `;
          uploadSubmitBtn.disabled = true;
        }, 1200);

        fetchDocumentList();
        fetchHealthTelemetry();

      } catch (err) {
        uploadStatusBox.classList.add('hidden');
        uploadSubmitBtn.disabled = false;
        showError(`Upload error: ${err.message}`);
      }
    });
  }

  async function fetchDocumentList() {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) return;
      const data = await res.json();
      renderDocumentList(data.documents || []);
    } catch (e) {
      console.error('Failed to load documents:', e);
    }
  }

  function renderDocumentList(docs) {
    if (!docListContainer) return;
    if (docCountBadge) docCountBadge.textContent = `${docs.length} Documents`;

    if (!docs || docs.length === 0) {
      docListContainer.innerHTML = `
        <div id="emptyKbState" class="empty-kb-state">
          <span class="empty-kb-icon">📖</span>
          <p>Knowledge base is empty.</p>
          <span class="empty-kb-sub">Upload a document above to provide context for RAG.</span>
        </div>
      `;
      return;
    }

    docListContainer.innerHTML = '';
    docs.forEach(doc => {
      const item = document.createElement('div');
      item.className = 'doc-item-card';

      const ext = (doc.file_type || 'txt').toLowerCase();
      let icon = '📝';
      if (ext === 'pdf') icon = '📕';
      else if (ext === 'md') icon = '📑';
      else if (ext === 'docx') icon = '📘';

      const sizeKb = doc.file_size ? `${(doc.file_size / 1024).toFixed(1)} KB` : '';

      item.innerHTML = `
        <div class="doc-info-wrap">
          <span class="doc-type-icon">${icon}</span>
          <div class="doc-details">
            <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
            <span class="doc-meta">
              <span class="doc-chunks-chip">${doc.number_of_chunks} chunk(s)</span>
              <span>•</span>
              <span>${sizeKb}</span>
            </span>
          </div>
        </div>
        <div class="doc-actions-row">
          <button type="button" class="doc-summarize-btn" data-doc-id="${doc.document_id}" data-filename="${escapeHtml(doc.filename)}" title="Summarize entire document">
            <span>✨</span> Summarize
          </button>
          <button type="button" class="doc-delete-btn" data-doc-id="${doc.document_id}" title="Remove document & vectors">
            ✕
          </button>
        </div>
      `;

      const sumBtn = item.querySelector('.doc-summarize-btn');
      sumBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        openSummarizeModal(doc.document_id, doc.filename);
      });

      const delBtn = item.querySelector('.doc-delete-btn');
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm(`Remove "${doc.filename}" and its vector embeddings?`)) {
          await deleteDocument(doc.document_id);
        }
      });

      docListContainer.appendChild(item);
    });
  }

  async function deleteDocument(docId) {
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      fetchDocumentList();
      fetchHealthTelemetry();
    } catch (err) {
      showError(`Failed to delete document: ${err.message}`);
    }
  }

  if (clearKbBtn) {
    clearKbBtn.addEventListener('click', async () => {
      if (confirm('Clear the entire knowledge base and all vector indexes?')) {
        try {
          const res = await fetch('/api/index/clear', { method: 'POST' });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          fetchDocumentList();
          fetchHealthTelemetry();
        } catch (err) {
          showError(`Failed to clear knowledge base: ${err.message}`);
        }
      }
    });
  }

  fetchDocumentList();

  // =========================================================================
  // 11. Complete Document Summarizer Modal
  // =========================================================================
  const summarizeModal = document.getElementById('summarizeModal');
  const closeSummarizeModal = document.getElementById('closeSummarizeModal');
  const summaryDocName = document.getElementById('summaryDocName');
  const startSummarizeBtn = document.getElementById('startSummarizeBtn');
  const summaryProgressBox = document.getElementById('summaryProgressBox');
  const summaryProgressText = document.getElementById('summaryProgressText');
  const summaryResultContainer = document.getElementById('summaryResultContainer');
  const summaryResultContent = document.getElementById('summaryResultContent');
  const copySummaryBtn = document.getElementById('copySummaryBtn');
  const styleOptionBtns = document.querySelectorAll('.style-option-btn');

  function openSummarizeModal(docId, filename) {
    activeSummarizeDocId = docId;
    if (summaryDocName) summaryDocName.textContent = filename;
    if (summaryResultContainer) summaryResultContainer.classList.add('hidden');
    if (summaryProgressBox) summaryProgressBox.classList.add('hidden');
    if (startSummarizeBtn) startSummarizeBtn.disabled = false;
    if (summarizeModal) summarizeModal.classList.remove('hidden');
  }

  function hideSummarizeModal() {
    if (summarizeModal) summarizeModal.classList.add('hidden');
    activeSummarizeDocId = null;
  }

  if (closeSummarizeModal) {
    closeSummarizeModal.addEventListener('click', hideSummarizeModal);
  }

  styleOptionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      styleOptionBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedSummarizeStyle = btn.getAttribute('data-style');
    });
  });

  if (startSummarizeBtn) {
    startSummarizeBtn.addEventListener('click', async () => {
      if (!activeSummarizeDocId) return;

      startSummarizeBtn.disabled = true;
      summaryProgressBox.classList.remove('hidden');
      summaryResultContainer.classList.add('hidden');
      summaryProgressText.textContent = 'Extracting document and analyzing sections...';

      try {
        const res = await fetch(`/api/documents/${activeSummarizeDocId}/summarize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ style: selectedSummarizeStyle })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP ${res.status}`);
        }

        const data = await res.json();
        summaryProgressBox.classList.add('hidden');
        summaryResultContainer.classList.remove('hidden');
        summaryResultContent.innerHTML = renderMarkdown(data.summary || '');

      } catch (err) {
        summaryProgressBox.classList.add('hidden');
        startSummarizeBtn.disabled = false;
        showError(`Summarization failed: ${err.message}`);
      }
    });
  }

  if (copySummaryBtn) {
    copySummaryBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(summaryResultContent.innerText || '');
        copySummaryBtn.textContent = '✓ Copied!';
        setTimeout(() => copySummaryBtn.textContent = 'Copy Summary', 1500);
      } catch (e) {
        copySummaryBtn.textContent = 'Failed';
      }
    });
  }

  // =========================================================================
  // 12. Settings Modal
  // =========================================================================
  const settingsModal = document.getElementById('settingsModal');
  const openSettingsBtn = document.getElementById('openSettingsBtn');
  const closeSettingsModal = document.getElementById('closeSettingsModal');
  const saveSettingsBtn = document.getElementById('saveSettingsBtn');
  const settingModelInput = document.getElementById('settingModelInput');
  const settingTempSlider = document.getElementById('settingTempSlider');
  const settingTempValue = document.getElementById('settingTempValue');
  const settingTopKSlider = document.getElementById('settingTopKSlider');
  const settingTopKValue = document.getElementById('settingTopKValue');
  const settingThresholdSlider = document.getElementById('settingThresholdSlider');
  const settingThresholdValue = document.getElementById('settingThresholdValue');

  if (openSettingsBtn) {
    openSettingsBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/settings');
        if (res.ok) {
          const s = await res.json();
          if (settingModelInput) settingModelInput.value = s.model || 'llama3:latest';
          if (settingTempSlider) {
            settingTempSlider.value = s.temperature || '0.7';
            settingTempValue.textContent = Number(settingTempSlider.value).toFixed(2);
          }
          if (settingTopKSlider) {
            settingTopKSlider.value = s.top_k || '3';
            settingTopKValue.textContent = settingTopKSlider.value;
          }
          if (settingThresholdSlider) {
            settingThresholdSlider.value = s.similarity_threshold || '0.20';
            settingThresholdValue.textContent = Number(settingThresholdSlider.value).toFixed(2);
          }
        }
      } catch (e) {}
      settingsModal.classList.remove('hidden');
    });
  }

  if (closeSettingsModal) {
    closeSettingsModal.addEventListener('click', () => settingsModal.classList.add('hidden'));
  }

  if (settingTempSlider) {
    settingTempSlider.addEventListener('input', (e) => {
      settingTempValue.textContent = Number(e.target.value).toFixed(2);
    });
  }

  if (settingTopKSlider) {
    settingTopKSlider.addEventListener('input', (e) => {
      settingTopKValue.textContent = e.target.value;
    });
  }

  if (settingThresholdSlider) {
    settingThresholdSlider.addEventListener('input', (e) => {
      settingThresholdValue.textContent = Number(e.target.value).toFixed(2);
    });
  }

  if (saveSettingsBtn) {
    saveSettingsBtn.addEventListener('click', async () => {
      const payload = {
        model: settingModelInput.value.trim(),
        temperature: settingTempSlider.value,
        top_k: settingTopKSlider.value,
        similarity_threshold: settingThresholdSlider.value
      };
      try {
        await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        settingsModal.classList.add('hidden');
        fetchHealthTelemetry();
      } catch (err) {
        showError(`Failed to save settings: ${err.message}`);
      }
    });
  }

  const unloadModelBtn = document.getElementById('unloadModelBtn');
  if (unloadModelBtn) {
    unloadModelBtn.addEventListener('click', async () => {
      unloadModelBtn.disabled = true;
      unloadModelBtn.textContent = 'Releasing...';
      try {
        const res = await fetch('/api/model/unload', { method: 'POST' });
        const data = await res.json();
        unloadModelBtn.textContent = data.status === 'success' ? '✓ VRAM Released' : 'Release Failed';
        setTimeout(() => {
          unloadModelBtn.disabled = false;
          unloadModelBtn.innerHTML = '<span>⚡</span> Release Llama 3 from GPU VRAM';
        }, 2000);
      } catch (e) {
        unloadModelBtn.disabled = false;
        unloadModelBtn.innerHTML = '<span>⚡</span> Release Llama 3 from GPU VRAM';
      }
    });
  }

  // =========================================================================
  // 13. Preserved Ultron v0.2 TinyStories PyTorch Laboratory
  // =========================================================================
  const promptInput = document.getElementById('promptInput');
  const generateBtn = document.getElementById('generateBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnLabel = document.getElementById('btnLabel');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const tempSlider = document.getElementById('tempSlider');
  const tempValue = document.getElementById('tempValue');
  const tokensSlider = document.getElementById('tokensSlider');
  const tokensValue = document.getElementById('tokensValue');
  const chatTranscript = document.getElementById('chatTranscript');
  let labWelcome = document.getElementById('welcomeState');

  const specModel = document.getElementById('specModel');
  const specVersion = document.getElementById('specVersion');
  const specVocab = document.getElementById('specVocab');
  const specContext = document.getElementById('specContext');
  const specParams = document.getElementById('specParams');
  const specDevice = document.getElementById('specDevice');
  const specArch = document.getElementById('specArch');

  if (tempSlider && tempValue) {
    tempSlider.addEventListener('input', (e) => {
      tempValue.textContent = Number(e.target.value).toFixed(2);
    });
  }

  if (tokensSlider && tokensValue) {
    tokensSlider.addEventListener('input', (e) => {
      tokensValue.textContent = e.target.value;
    });
  }

  async function fetchModelInfo() {
    try {
      const res = await fetch('/api/info');
      if (!res.ok) return;
      const data = await res.json();
      if (specModel) specModel.textContent = data.model;
      if (specVersion) specVersion.textContent = data.version;
      if (specVocab) specVocab.textContent = Number(data.vocab_size).toLocaleString();
      if (specContext) specContext.textContent = data.context;
      if (specParams) specParams.textContent = Number(data.parameters).toLocaleString();
      if (specDevice) specDevice.textContent = data.device;
      if (specArch) specArch.textContent = `Structure: ${data.architecture} • ${data.context} Context`;
    } catch (e) {
      console.error('Failed to load v0.2 info:', e);
    }
  }

  fetchModelInfo();

  // Lab suggestions
  const labChips = document.querySelectorAll('.lab-chip');
  labChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const p = chip.getAttribute('data-prompt');
      if (promptInput) {
        promptInput.value = p;
        handleLabGenerate();
      }
    });
  });

  async function handleLabGenerate() {
    const prompt = (promptInput.value || '').trim();
    if (!prompt) return;

    generateBtn.disabled = true;
    if (btnSpinner) btnSpinner.classList.remove('hidden');
    if (btnLabel) btnLabel.classList.add('hidden');

    if (labWelcome && labWelcome.parentNode) {
      labWelcome.remove();
    }

    // User message in lab
    const userRow = document.createElement('div');
    userRow.className = 'message-bubble-row user-row';
    userRow.innerHTML = `
      <div class="message-meta"><span>Prompt</span></div>
      <div class="user-bubble">${escapeHtml(prompt)}</div>
    `;
    chatTranscript.appendChild(userRow);
    chatTranscript.scrollTop = chatTranscript.scrollHeight;

    promptInput.value = '';

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          temperature: parseFloat(tempSlider.value),
          max_new_tokens: parseInt(tokensSlider.value)
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }

      const data = await res.json();

      const assistantRow = document.createElement('div');
      assistantRow.className = 'message-bubble-row ultron-row';
      assistantRow.innerHTML = `
        <div class="ultron-bubble">
          <div class="bubble-head">
            <div class="agent-badge">
              <span class="agent-avatar-mini" style="background: #E06449;">⚡</span>
              <span>Ultron v0.2 PyTorch</span>
            </div>
            <span class="telemetry-chip">${data.tokens_generated} tokens in ${data.latency_ms}ms (${data.tokens_per_sec} tok/s)</span>
          </div>
          <div class="generated-text-content">${escapeHtml(data.text)}</div>
        </div>
      `;
      chatTranscript.appendChild(assistantRow);
      chatTranscript.scrollTop = chatTranscript.scrollHeight;

    } catch (err) {
      showError(`v0.2 generation error: ${err.message}`);
    } finally {
      generateBtn.disabled = false;
      if (btnSpinner) btnSpinner.classList.add('hidden');
      if (btnLabel) btnLabel.classList.remove('hidden');
    }
  }

  if (generateBtn) {
    generateBtn.addEventListener('click', handleLabGenerate);
  }

  if (promptInput) {
    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleLabGenerate();
      }
    });
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      chatTranscript.innerHTML = '';
      if (labWelcome) chatTranscript.appendChild(labWelcome);
    });
  }

  // Initial load
  fetchConversations();
});
