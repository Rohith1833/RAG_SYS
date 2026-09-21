/**
 * Ultron AI Assistant & Laboratory — Frontend Controller
 * Integrates:
 * 1. Local RAG Assistant (Semantic Retrieval + llama3:latest + all-MiniLM-L6-v2)
 * 2. Ultron v0.2 TinyStories PyTorch Laboratory (Preserved Baseline)
 */

document.addEventListener('DOMContentLoaded', () => {

  // =========================================================================
  // 1. Navigation & Tab Switcher
  // =========================================================================
  const tabRagBtn = document.getElementById('tabRagBtn');
  const tabLabBtn = document.getElementById('tabLabBtn');
  const ragView = document.getElementById('ragView');
  const labView = document.getElementById('labView');

  function switchTab(target) {
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
    tabRagBtn.addEventListener('click', () => switchTab('ragView'));
    tabLabBtn.addEventListener('click', () => switchTab('labView'));
  }

  // =========================================================================
  // 2. Global Notifications & Status
  // =========================================================================
  const statusBadge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
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
  // 3. System Telemetry & Health Checks
  // =========================================================================
  const telemetryModel = document.getElementById('telemetryModel');
  const telemetryOllama = document.getElementById('telemetryOllama');
  const telemetryEmbedder = document.getElementById('telemetryEmbedder');
  const telemetryDim = document.getElementById('telemetryDim');
  const telemetryDevice = document.getElementById('telemetryDevice');
  const telemetryChunks = document.getElementById('telemetryChunks');
  const docCountBadge = document.getElementById('docCountBadge');

  async function fetchHealthTelemetry() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error(`Health HTTP ${res.status}`);
      const data = await res.json();

      if (data.ollama) {
        statusBadge.className = 'status-pill clay-pill online';
        statusText.textContent = `Online • ${(data.device || 'CUDA').toUpperCase()} (${data.model || 'llama3:latest'})`;
        hideError();
      } else {
        statusBadge.className = 'status-pill clay-pill offline-warning';
        statusText.textContent = 'Ollama Offline • Ingestion Ready';
      }

      if (telemetryModel) telemetryModel.textContent = data.model || 'llama3:latest';
      if (telemetryOllama) {
        telemetryOllama.textContent = data.ollama ? 'Connected (Local)' : 'Offline (Generation Unavailable)';
        telemetryOllama.style.color = data.ollama ? 'var(--status-green)' : '#D97706';
      }
      if (telemetryEmbedder) telemetryEmbedder.textContent = data.embedding_model || 'all-MiniLM-L6-v2';
      if (telemetryDim) telemetryDim.textContent = data.embedding_dimension || '384';
      if (telemetryDevice) telemetryDevice.textContent = data.device ? data.device.toUpperCase() : 'Detecting...';
      if (telemetryChunks) telemetryChunks.textContent = `${data.chunks || 0} chunk(s) in memory`;
      if (docCountBadge) docCountBadge.textContent = `${data.documents || 0} Documents`;

    } catch (err) {
      statusBadge.className = 'status-pill clay-pill error';
      statusText.textContent = 'Backend Error';
    }
  }

  // Initial health check
  fetchHealthTelemetry();

  // =========================================================================
  // 4. Local RAG Document Ingestion & Management
  // =========================================================================
  const uploadForm = document.getElementById('uploadForm');
  const fileInput = document.getElementById('fileInput');
  const dropzoneContent = document.getElementById('dropzoneContent');
  const uploadSubmitBtn = document.getElementById('uploadSubmitBtn');
  const uploadStatusBox = document.getElementById('uploadStatusBox');
  const uploadStatusText = document.getElementById('uploadStatusText');
  const uploadProgressBar = document.getElementById('uploadProgressBar');
  const docListContainer = document.getElementById('docListContainer');
  const emptyKbState = document.getElementById('emptyKbState');
  const clearKbBtn = document.getElementById('clearKbBtn');

  let selectedFile = null;

  // File selection
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        setFileToUpload(files[0]);
      }
    });
  }

  // Drag & drop support
  if (uploadForm) {
    ['dragenter', 'dragover'].forEach(eventName => {
      uploadForm.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadForm.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      uploadForm.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadForm.classList.remove('drag-over');
      });
    });

    uploadForm.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        setFileToUpload(dt.files[0]);
      }
    });
  }

  function setFileToUpload(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!['.txt', '.md', '.pdf'].includes(ext)) {
      showError(`Unsupported file format '${ext}'. Please select a .txt, .md, or .pdf file.`);
      return;
    }

    selectedFile = file;
    hideError();

    dropzoneContent.innerHTML = `
      <span class="upload-icon">📄</span>
      <span class="dropzone-text">${file.name}</span>
      <span class="supported-formats">${(file.size / 1024).toFixed(1)} KB • Ready to index</span>
    `;

    uploadSubmitBtn.disabled = false;
  }

  // Handle document upload
  if (uploadSubmitBtn) {
    uploadSubmitBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      hideError();
      uploadSubmitBtn.disabled = true;
      uploadStatusBox.classList.remove('hidden');

      // Staged progress state helper
      const updateProgress = (text, percent) => {
        uploadStatusText.textContent = text;
        uploadProgressBar.style.width = `${percent}%`;
      };

      updateProgress('Uploading file to local store...', 15);

      const formData = new FormData();
      formData.append('file', selectedFile);

      // Staged progress updates to keep the user informed during large document indexing
      const timer1 = setTimeout(() => updateProgress('Parsing document text & pages (DocumentParser)...', 35), 400);
      const timer2 = setTimeout(() => updateProgress('Generating dense embeddings (all-MiniLM-L6-v2)...', 65), 1200);
      const timer3 = setTimeout(() => updateProgress('Storing in NumPy VectorStore & persisting index...', 85), 2500);

      try {
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });

        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || `Upload failed with status ${response.status}`);
        }

        updateProgress(`Indexed ${data.chunks_indexed || 'all'} chunk(s) successfully!`, 100);

        setTimeout(() => {
          uploadStatusBox.classList.add('hidden');
          uploadProgressBar.style.width = '0%';
          selectedFile = null;
          fileInput.value = '';
          dropzoneContent.innerHTML = `
            <span class="upload-icon">📁</span>
            <span class="dropzone-text">Click or drag file to upload</span>
            <span class="supported-formats">Supported: <strong>.TXT</strong> • <strong>.MD</strong> • <strong>.PDF</strong></span>
          `;
          uploadSubmitBtn.disabled = true;
        }, 1200);

        // Refresh Knowledge Base and Telemetry
        fetchDocumentList();
        fetchHealthTelemetry();

      } catch (err) {
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
        uploadStatusBox.classList.add('hidden');
        uploadSubmitBtn.disabled = false;
        showError(`Upload error: ${err.message}`);
      }
    });
  }

  // Fetch document list
  async function fetchDocumentList() {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const docs = data.documents || [];

      renderDocumentList(docs);
      if (docCountBadge) docCountBadge.textContent = `${docs.length} Documents`;

    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  }

  function renderDocumentList(docs) {
    if (!docListContainer) return;

    if (!docs || docs.length === 0) {
      docListContainer.innerHTML = `
        <div id="emptyKbState" class="empty-kb-state">
          <span class="empty-kb-icon">📖</span>
          <p>Your knowledge base is empty.</p>
          <span class="empty-kb-sub">Upload a document above to give Ultron local context.</span>
        </div>
      `;
      return;
    }

    docListContainer.innerHTML = '';

    docs.forEach((doc) => {
      const item = document.createElement('div');
      item.className = 'doc-item-card';

      const ext = doc.file_type ? doc.file_type.toLowerCase() : 'txt';
      let icon = '📝';
      if (ext === 'pdf') icon = '📕';
      else if (ext === 'md') icon = '📑';

      item.innerHTML = `
        <div class="doc-info-wrap">
          <span class="doc-type-icon">${icon}</span>
          <div class="doc-details">
            <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
            <span class="doc-meta">
              <span class="doc-chunks-chip">${doc.number_of_chunks} chunk(s)</span>
              <span>•</span>
              <span>${doc.uploaded_at ? doc.uploaded_at.split(' ')[0] : 'Indexed'}</span>
            </span>
          </div>
        </div>
        <button type="button" class="doc-delete-btn" data-doc-id="${doc.document_id}" title="Remove document from knowledge base">
          ✕
        </button>
      `;

      const deleteBtn = item.querySelector('.doc-delete-btn');
      deleteBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const docId = deleteBtn.getAttribute('data-doc-id');
        if (confirm(`Remove "${doc.filename}" from the local knowledge base?`)) {
          await deleteDocument(docId);
        }
      });

      docListContainer.appendChild(item);
    });
  }

  async function deleteDocument(docId) {
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `HTTP ${res.status}`);
      }
      fetchDocumentList();
      fetchHealthTelemetry();
    } catch (err) {
      showError(`Failed to delete document: ${err.message}`);
    }
  }

  // Clear Knowledge Base
  if (clearKbBtn) {
    clearKbBtn.addEventListener('click', async () => {
      if (confirm('Are you sure you want to clear the entire knowledge base and vector store?')) {
        try {
          const res = await fetch('/api/index/clear', { method: 'POST' });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          fetchDocumentList();
          fetchHealthTelemetry();
        } catch (err) {
          showError(`Failed to clear index: ${err.message}`);
        }
      }
    });
  }

  // Fetch initial documents on load
  fetchDocumentList();

  // =========================================================================
  // 5. Local RAG Chat Interaction
  // =========================================================================
  const ragTranscript = document.getElementById('ragTranscript');
  const ragWelcomeState = document.getElementById('ragWelcomeState');
  const ragPromptInput = document.getElementById('ragPromptInput');
  const ragSendBtn = document.getElementById('ragSendBtn');
  const ragBtnSpinner = document.getElementById('ragBtnSpinner');
  const ragBtnLabel = document.getElementById('ragBtnLabel');
  const clearRagChatBtn = document.getElementById('clearRagChatBtn');

  let isRagQuerying = false;

  function scrollToRagBottom() {
    if (ragTranscript) {
      ragTranscript.scrollTop = ragTranscript.scrollHeight;
    }
  }

  function appendRagUserMessage(text) {
    if (ragWelcomeState && ragWelcomeState.parentNode) {
      ragWelcomeState.remove();
    }

    const row = document.createElement('div');
    row.className = 'message-bubble-row user-row';
    row.innerHTML = `
      <div class="message-meta"><span>You</span></div>
      <div class="user-bubble">${escapeHtml(text)}</div>
    `;
    ragTranscript.appendChild(row);
    scrollToRagBottom();
  }

  function createRagGeneratingBubble() {
    const row = document.createElement('div');
    row.id = 'ragGeneratingIndicator';
    row.className = 'message-bubble-row ultron-row';
    row.innerHTML = `
      <div class="ultron-bubble generating-bubble">
        <div class="pulse-spinner"></div>
        <span class="generating-text">Searching vector store & reasoning with Llama 3...</span>
      </div>
    `;
    ragTranscript.appendChild(row);
    scrollToRagBottom();
    return row;
  }

  function appendRagAssistantMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-bubble-row ultron-row';

    const bubble = document.createElement('div');
    bubble.className = 'ultron-bubble';

    // Header
    const head = document.createElement('div');
    head.className = 'bubble-head';

    const isOfflineMode = (data.generation_available === false || data.ollama_available === false);

    const badge = document.createElement('div');
    badge.className = 'agent-badge';
    if (isOfflineMode) {
      badge.style.color = '#B45309';
      badge.innerHTML = `
        <span class="agent-avatar-mini" style="background: #D97706;">✦</span>
        <span>Ultron Semantic Retrieval (Ollama Offline)</span>
      `;
    } else {
      badge.innerHTML = `
        <span class="agent-avatar-mini">✦</span>
        <span>Ultron Local Assistant</span>
      `;
    }

    const telemetry = document.createElement('span');
    telemetry.className = 'telemetry-chip';
    const latency = data.retrieval_ms ? `Retrieved in ${data.retrieval_ms}ms` : 'Local Inference';
    telemetry.textContent = latency;

    head.appendChild(badge);
    head.appendChild(telemetry);

    // Answer Body
    const content = document.createElement('div');
    content.className = 'generated-text-content';
    content.textContent = data.answer;

    bubble.appendChild(head);
    bubble.appendChild(content);

    // Render Sources Used (Step 10 Requirement)
    if (data.sources && data.sources.length > 0) {
      const sourcesCard = document.createElement('div');
      sourcesCard.className = 'sources-card';

      const sourcesHead = document.createElement('div');
      sourcesHead.className = 'sources-head';
      sourcesHead.innerHTML = `
        <span class="sources-title">
          <span>📚</span> Sources Used (${data.sources.length})
        </span>
      `;
      sourcesCard.appendChild(sourcesHead);

      const sourcesList = document.createElement('div');
      sourcesList.className = 'sources-list';

      data.sources.forEach((src) => {
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

    // Footer with Copy Action
    const footer = document.createElement('div');
    footer.className = 'bubble-footer';

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'action-pill-btn';
    copyBtn.textContent = 'Copy answer';
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(data.answer);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => copyBtn.textContent = 'Copy answer', 1800);
      } catch (e) {
        copyBtn.textContent = 'Failed to copy';
      }
    });

    footer.appendChild(copyBtn);
    bubble.appendChild(footer);

    row.appendChild(bubble);
    ragTranscript.appendChild(row);
    scrollToRagBottom();
  }

  async function handleRagQuery() {
    if (isRagQuerying) return;

    const query = ragPromptInput.value.trim();
    if (!query) {
      ragPromptInput.focus();
      return;
    }

    hideError();
    isRagQuerying = true;
    ragSendBtn.disabled = true;
    ragBtnSpinner.classList.remove('hidden');
    ragBtnLabel.innerHTML = '<span>Answering...</span>';

    appendRagUserMessage(query);
    const indicator = createRagGeneratingBubble();
    ragPromptInput.value = '';

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query })
      });

      const result = await response.json();
      if (indicator && indicator.parentNode) indicator.remove();

      if (!response.ok) {
        throw new Error(result.error || `Server error ${response.status}`);
      }

      appendRagAssistantMessage(result);

    } catch (err) {
      if (indicator && indicator.parentNode) indicator.remove();
      showError(err.message || 'Error occurred while answering query.');

      const errRow = document.createElement('div');
      errRow.className = 'message-bubble-row ultron-row';
      errRow.innerHTML = `
        <div class="ultron-bubble" style="border-color: #FCA5A5; background: #FFF5F5;">
          <div class="bubble-head">
            <div class="agent-badge" style="color: #DC2626;">
              <span class="agent-avatar-mini" style="background: #DC2626;">✕</span>
              <span>Assistant Error</span>
            </div>
          </div>
          <div class="generated-text-content" style="color: #991B1B;">${escapeHtml(err.message)}</div>
        </div>
      `;
      ragTranscript.appendChild(errRow);
      scrollToRagBottom();

    } finally {
      isRagQuerying = false;
      ragSendBtn.disabled = false;
      ragBtnSpinner.classList.add('hidden');
      ragBtnLabel.innerHTML = '<span class="action-icon">▶</span> Ask Ultron';
      ragPromptInput.focus();
    }
  }

  if (ragSendBtn) {
    ragSendBtn.addEventListener('click', handleRagQuery);
  }

  if (ragPromptInput) {
    ragPromptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleRagQuery();
      }
    });
  }

  if (clearRagChatBtn) {
    clearRagChatBtn.addEventListener('click', () => {
      ragTranscript.innerHTML = '';
      if (ragWelcomeState) ragTranscript.appendChild(ragWelcomeState);
      hideError();
      ragPromptInput.focus();
    });
  }

  // RAG Suggestion Chips
  document.querySelectorAll('.rag-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt && ragPromptInput) {
        ragPromptInput.value = prompt;
        ragPromptInput.focus();
      }
    });
  });

  // =========================================================================
  // 6. v0.2 TinyStories PyTorch Laboratory (Preserved Baseline)
  // =========================================================================
  const labTranscript = document.getElementById('chatTranscript');
  const labWelcomeState = document.getElementById('welcomeState');
  const promptInput = document.getElementById('promptInput');
  const generateBtn = document.getElementById('generateBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnLabel = document.getElementById('btnLabel');
  const tempSlider = document.getElementById('tempSlider');
  const tempValue = document.getElementById('tempValue');
  const tokensSlider = document.getElementById('tokensSlider');
  const tokensValue = document.getElementById('tokensValue');
  const clearChatBtn = document.getElementById('clearChatBtn');

  const specModel = document.getElementById('specModel');
  const specVersion = document.getElementById('specVersion');
  const specVocab = document.getElementById('specVocab');
  const specContext = document.getElementById('specContext');
  const specParams = document.getElementById('specParams');
  const specDevice = document.getElementById('specDevice');
  const specArch = document.getElementById('specArch');

  let isGenerating = false;

  if (tempSlider && tempValue) {
    tempSlider.addEventListener('input', (e) => {
      tempValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  if (tokensSlider && tokensValue) {
    tokensSlider.addEventListener('input', (e) => {
      tokensValue.textContent = e.target.value;
    });
  }

  document.querySelectorAll('.lab-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt && promptInput) {
        promptInput.value = prompt;
        promptInput.focus();
      }
    });
  });

  async function fetchLabModelInfo() {
    try {
      const res = await fetch('/api/info');
      if (!res.ok) return;
      const data = await res.json();

      if (specModel) specModel.textContent = data.model || 'Ultron';
      if (specVersion) specVersion.textContent = data.version || 'v0.2';
      if (specVocab) specVocab.textContent = data.vocab_size ? Number(data.vocab_size).toLocaleString() : '2,000';
      if (specContext) specContext.textContent = data.context ? Number(data.context).toLocaleString() : '128';
      if (specParams) specParams.textContent = data.parameters ? Number(data.parameters).toLocaleString() : '1,323,728';
      if (specDevice) specDevice.textContent = data.device || 'Detecting...';
      if (specArch && data.architecture) {
        specArch.textContent = `Structure: ${data.architecture} • ${data.context} Context`;
      }
    } catch (e) {
      console.warn('Failed to load v0.2 lab info:', e);
    }
  }

  fetchLabModelInfo();

  function appendLabUserMessage(text) {
    if (labWelcomeState && labWelcomeState.parentNode) labWelcomeState.remove();

    const row = document.createElement('div');
    row.className = 'message-bubble-row user-row';
    row.innerHTML = `
      <div class="message-meta"><span>You</span></div>
      <div class="user-bubble">${escapeHtml(text)}</div>
    `;
    labTranscript.appendChild(row);
    labTranscript.scrollTop = labTranscript.scrollHeight;
  }

  function createLabGeneratingBubble() {
    const row = document.createElement('div');
    row.className = 'message-bubble-row ultron-row';
    row.innerHTML = `
      <div class="ultron-bubble generating-bubble">
        <div class="pulse-spinner"></div>
        <span class="generating-text">Sampling tokens from Ultron v0.2 (TinyStories)...</span>
      </div>
    `;
    labTranscript.appendChild(row);
    labTranscript.scrollTop = labTranscript.scrollHeight;
    return row;
  }

  function appendLabAssistantMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-bubble-row ultron-row';

    const bubble = document.createElement('div');
    bubble.className = 'ultron-bubble';

    const head = document.createElement('div');
    head.className = 'bubble-head';
    head.innerHTML = `
      <div class="agent-badge">
        <span class="agent-avatar-mini">✦</span>
        <span>Ultron v0.2</span>
      </div>
      <span class="telemetry-chip">
        ${data.tokens_generated || 100} tokens • ${data.latency_ms || 0}ms • ${data.tokens_per_sec || 0} tok/s
      </span>
    `;

    const content = document.createElement('div');
    content.className = 'generated-text-content';
    content.textContent = data.text;

    bubble.appendChild(head);
    bubble.appendChild(content);
    row.appendChild(bubble);
    labTranscript.appendChild(row);
    labTranscript.scrollTop = labTranscript.scrollHeight;
  }

  async function handleLabGenerate() {
    if (isGenerating) return;

    const rawPrompt = promptInput.value.trim();
    if (!rawPrompt) {
      promptInput.focus();
      return;
    }

    isGenerating = true;
    generateBtn.disabled = true;
    btnSpinner.classList.remove('hidden');
    btnLabel.innerHTML = '<span>Generating...</span>';

    appendLabUserMessage(rawPrompt);
    const indicator = createLabGeneratingBubble();
    promptInput.value = '';

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: rawPrompt,
          temperature: parseFloat(tempSlider.value) || 0.8,
          max_new_tokens: parseInt(tokensSlider.value, 10) || 100
        })
      });

      const result = await response.json();
      if (indicator && indicator.parentNode) indicator.remove();

      if (!response.ok) {
        throw new Error(result.error || `Error ${response.status}`);
      }

      appendLabAssistantMessage(result);

    } catch (err) {
      if (indicator && indicator.parentNode) indicator.remove();
      showError(err.message || 'Error occurred during generation.');
    } finally {
      isGenerating = false;
      generateBtn.disabled = false;
      btnSpinner.classList.add('hidden');
      btnLabel.innerHTML = '<span class="action-icon">▶</span> Generate';
      promptInput.focus();
    }
  }

  if (generateBtn) generateBtn.addEventListener('click', handleLabGenerate);

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
      labTranscript.innerHTML = '';
      if (labWelcomeState) labTranscript.appendChild(labWelcomeState);
      promptInput.focus();
    });
  }

  // Escape HTML helper
  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
});
