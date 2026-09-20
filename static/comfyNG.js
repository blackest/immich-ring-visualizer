/**
 * comfyNG.js -- the "ComfyUI" view (bottom-bar ComfyUI task).
 *
 * Pick a PNG that ComfyUI generated (it embeds the exact workflow that
 * made it), get a dynamic form for every plain value in that workflow,
 * edit whatever you want, Generate, get one result image back. Wired to
 * /api/ng/comfy/* (routes/comfyNG.py), which does the actual PNG-
 * metadata extraction and proxies the run to a real ComfyUI server (see
 * configNG.get_comfyui_base_url -- editable in the settings modal under
 * "Addresses").
 *
 * No project/character concept, same as hdmultiNG.js -- a workflow here
 * doesn't belong to any tab. Loaded after hdmultiNG.js and before
 * bootstrapWiringNG.js (which fires the first ProjectManager.render(),
 * which calls ComfyNG.sync()).
 */
(function () {
  "use strict";

  var inited = false;
  var els = {};
  var extractId = null;
  var fields = []; // [{nodeId, classType, inputName, value, valueType}] -- the live form; a Generate click snapshots this, see snapshotFields
  var currentWorkflowLabel = null; // whatever's currently loaded, just for queue-row titles (a PNG's filename, or a saved workflow's name)
  var queue = []; // [{localId, jobId, status: "submitting"|"running"|"done"|"error", label, error, resultUrl, savedName}]
  var localSeq = 0;
  var pollTimer = null;
  var POLL_MS = 2000;
  var activeImageField = null; // the field object last armed via "Pick from server list", or null
  var activeImageWrapEl = null; // its dropbox element, for the highlight class -- cleared together with activeImageField

  function $(id) {
    return document.getElementById(id);
  }

  function refreshEls() {
    els.pane = $("ng-comfy-pane");
    els.main = $("ng-comfy-main");
    els.controlsPane = $("ng-controls-pane");
    els.pngFile = $("ng-comfy-png-file");
    els.pngFileBtn = $("ng-comfy-png-file-btn");
    els.pngPaste = $("ng-comfy-png-paste");
    els.extractStatus = $("ng-comfy-extract-status");
    els.fieldsWrap = $("ng-comfy-fields");
    els.actions = $("ng-comfy-actions");
    els.generateBtn = $("ng-comfy-generate-btn");
    els.workflowList = $("ng-comfy-workflow-list");
    els.serverImages = $("ng-comfy-server-images");
    els.imagesTabInput = $("ng-comfy-images-tab-input");
    els.imagesTabOutput = $("ng-comfy-images-tab-output");
    els.serverImagesFilter = $("ng-comfy-server-images-filter");
    els.serverImagesPager = $("ng-comfy-server-images-pager");
    els.serverImagesPrev = $("ng-comfy-server-images-prev");
    els.serverImagesNext = $("ng-comfy-server-images-next");
    els.serverImagesPageLabel = $("ng-comfy-server-images-page-label");
    els.queueCount = $("ng-comfy-queue-count");
    els.queueEmpty = $("ng-comfy-queue-empty");
    els.queue = $("ng-comfy-queue");
  }

  function setExtractStatus(msg) {
    if (els.extractStatus) els.extractStatus.textContent = msg || "";
  }

  // A LoadImage node's `image` field (valueType "image", see
  // routes/comfyNG.py's _is_image_field) -- the img2img/img2video entry
  // point. f.value starts out as whatever filename the saved/loaded
  // workflow already had wired on ComfyUI's own server; this previews
  // that (via the /view proxy) and lets you replace it with a pasted or
  // picked photo. Same paste-box-doubles-as-preview pattern and
  // -filled/pendingUrl handling as Generate's own ref picker
  // (generateNG.js's renderRefPreview) rather than a bespoke one here.
  // The replacement isn't uploaded here -- startGenerate uploads every
  // field with a pending file first and only then swaps f.value to the
  // filename ComfyUI hands back, so nothing reaches ComfyUI until you
  // actually click Generate.
  function buildImageField(f) {
    var wrap = document.createElement("div");
    wrap.className = "ng-gen-ref-paste ng-comfy-image-field";
    wrap.tabIndex = 0;
    wrap.title = "Click or drop an image here, or paste (Cmd+V) -- iPad has no Cmd+V, so click opens its own photo picker.";

    // Click-to-browse opens the system file/photo picker -- on iPad
    // there's no real Ctrl/Cmd+V for pasting an image, and no drag
    // source to drop from either, so the box itself has to be a usable
    // target, not just a paste catcher. Shared with diskBtn below (kept
    // as a second, more discoverable way to reach the same picker).
    function browseForFile() {
      var fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = "image/*";
      fileInput.addEventListener("change", function () {
        if (fileInput.files && fileInput.files[0]) setPending(fileInput.files[0]);
      });
      fileInput.click();
    }

    var clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    clearBtn.textContent = "Clear replacement";
    clearBtn.addEventListener("click", function () {
      if (f._pendingUrl) URL.revokeObjectURL(f._pendingUrl);
      f._pendingFile = null;
      f._pendingUrl = null;
      renderPreview();
    });

    var diskBtn = document.createElement("button");
    diskBtn.type = "button";
    diskBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    diskBtn.textContent = "+ choose from disk…";
    diskBtn.addEventListener("click", browseForFile);

    // For multi-reference workflows (identity ref / scene ref / clothing
    // ref, say) whose source images live on whatever machine ComfyUI
    // itself runs on rather than this device -- arms this field as the
    // target, then a click in the "ComfyUI's own input images" gallery
    // in the left rail (renderServerImages below) fills it in directly,
    // no upload needed since the file's already there. Toggles off on a
    // second click, same as the gallery cancelling on a successful pick.
    var serverBtn = document.createElement("button");
    serverBtn.type = "button";
    serverBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    serverBtn.textContent = "Pick from server list ←";
    serverBtn.title = "Then choose an image from ComfyUI's own input folder in the left column.";
    serverBtn.addEventListener("click", function () {
      setActiveImageField(activeImageField === f ? null : f, wrap);
    });

    var btnRow = document.createElement("div");
    btnRow.className = "ng-hdmulti-ref-slot-btns";
    btnRow.appendChild(diskBtn);
    btnRow.appendChild(serverBtn);
    btnRow.appendChild(clearBtn);

    function setPending(file) {
      if (f._pendingUrl) URL.revokeObjectURL(f._pendingUrl);
      f._pendingFile = file;
      f._pendingUrl = URL.createObjectURL(file);
      renderPreview();
    }

    // Called from pickServerImage when this field is the active target --
    // the filename is already valid on ComfyUI's own server, so it goes
    // straight into f.value like an original wired-in value would, no
    // upload/pendingFile round trip like the disk/paste path needs.
    f._applyServerFilename = function (filename) {
      if (f._pendingUrl) URL.revokeObjectURL(f._pendingUrl);
      f._pendingFile = null;
      f._pendingUrl = null;
      f.value = filename;
      renderPreview();
    };

    function currentValueUrl() {
      if (!f.value) return null;
      // A value picked from the output/temp gallery carries ComfyUI's
      // own "name [output]" annotation (folder_paths.get_annotated_
      // filepath) -- strip it back off to preview it, same type the
      // annotation names rather than always assuming input.
      var value = String(f.value);
      var type = "input";
      var m = /^(.*) \[(input|output|temp)\]$/.exec(value);
      if (m) {
        value = m[1];
        type = m[2];
      }
      var parts = value.split("/");
      var filename = parts.pop();
      var subfolder = parts.join("/");
      return "/api/ng/comfy/view?filename=" + encodeURIComponent(filename) +
        "&subfolder=" + encodeURIComponent(subfolder) + "&type=" + type;
    }

    // Rebuilds wrap's content from scratch each time, same as Generate's
    // renderRefPreview -- a pending replacement wins over the original
    // wired value, which wins over the empty-paste placeholder. The
    // original value's preview can 404 (ComfyUI may have long since
    // cleaned up that input file) -- onerror just falls back to the
    // plain filename text rather than a broken-image icon.
    function renderPreview() {
      var url = f._pendingUrl || currentValueUrl();
      if (url) {
        wrap.innerHTML = "";
        var img = document.createElement("img");
        img.alt = "";
        img.src = url;
        img.addEventListener("error", function () {
          wrap.classList.remove("ng-gen-ref-paste-filled");
          wrap.textContent = "Currently: " + f.value + " (preview unavailable)";
        });
        wrap.appendChild(img);
        wrap.classList.add("ng-gen-ref-paste-filled");
      } else {
        wrap.classList.remove("ng-gen-ref-paste-filled");
        wrap.textContent = "Click or drop an image here (or paste with Ctrl+V)";
      }
      clearBtn.style.display = f._pendingFile ? "" : "none";
    }
    renderPreview();

    wrap.addEventListener("paste", function (e) {
      var items = (e.clipboardData && e.clipboardData.items) || [];
      for (var i = 0; i < items.length; i++) {
        if (items[i].kind === "file" && /^image\//.test(items[i].type)) {
          e.preventDefault();
          setPending(items[i].getAsFile());
          return;
        }
      }
    });

    // iPad has no reliable Ctrl/Cmd+V for an image and no drag source
    // to drop from within the app itself, but Safari/iPadOS *does*
    // support dropping a photo dragged in from the Files/Photos app --
    // and a plain click opens the same native picker as "+ choose from
    // disk" above, which is the actually-usable path there.
    wrap.addEventListener("click", browseForFile);
    wrap.addEventListener("dragover", function (e) {
      e.preventDefault();
    });
    wrap.addEventListener("drop", function (e) {
      e.preventDefault();
      var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file && /^image\//.test(file.type)) setPending(file);
    });

    var container = document.createElement("div");
    container.appendChild(wrap);
    container.appendChild(btnRow);
    return container;
  }

  // Grouped by node (a node's fields shown together under one heading)
  // rather than one flat list -- the same node_id/class_type pairing a
  // ComfyUI graph itself uses, so e.g. all of KSampler's seed/steps/cfg
  // read as belonging together instead of an unordered pile of inputs.
  function renderFields() {
    els.fieldsWrap.innerHTML = "";
    var byNode = {};
    var order = [];
    fields.forEach(function (f) {
      if (!byNode[f.nodeId]) {
        byNode[f.nodeId] = [];
        order.push(f.nodeId);
      }
      byNode[f.nodeId].push(f);
    });

    order.forEach(function (nodeId) {
      var nodeFields = byNode[nodeId];
      var group = document.createElement("div");
      group.className = "ng-comfy-field-group";

      var heading = document.createElement("div");
      heading.className = "ng-gen-label";
      // A saved workflow's PNG carries node titles ("Load Image-insert
      // reolace" etc) via the extra 'workflow' metadata chunk (see
      // routes/comfyNG.py's _extract_node_titles) -- shown when present
      // so two same-class_type nodes (e.g. two LoadImage boxes) read as
      // distinct instead of both just "LoadImage (N)".
      heading.textContent = nodeFields[0].title
        ? nodeFields[0].title + " — " + nodeFields[0].classType + " (" + nodeId + ")"
        : nodeFields[0].classType + " (" + nodeId + ")";
      group.appendChild(heading);

      nodeFields.forEach(function (f) {
        var row = document.createElement("div");
        row.className = "ng-comfy-field-row";

        var label = document.createElement("label");
        label.textContent = f.inputName;
        label.className = "ng-comfy-field-label";
        row.appendChild(label);

        var input;
        if (f.valueType === "image") {
          input = buildImageField(f);
        } else if (f.valueType === "bool") {
          input = document.createElement("input");
          input.type = "checkbox";
          input.checked = !!f.value;
          input.addEventListener("change", function () {
            f.value = input.checked;
          });
        } else if (typeof f.value === "string" && f.value.length > 60) {
          input = document.createElement("textarea");
          input.rows = 4;
          input.className = "ng-comfy-field-textarea";
          input.value = f.value;
          input.addEventListener("input", function () {
            f.value = input.value;
          });
        } else {
          input = document.createElement("input");
          input.type = f.valueType === "number" ? "number" : "text";
          if (f.valueType === "number" && !Number.isInteger(f.value)) input.step = "any";
          input.className = "ng-gen-input";
          input.value = f.value;
          input.addEventListener("input", function () {
            f.value = f.valueType === "number" ? Number(input.value) : input.value;
          });
        }
        row.appendChild(input);
        group.appendChild(row);
      });

      els.fieldsWrap.appendChild(group);
    });

    els.actions.style.display = fields.length ? "" : "none";
  }

  // Shared by every extraction source (upload, paste, saved-workflow
  // library, and a server-images gallery pick) -- each just fetches
  // {extractId, fields} its own way and hands it here to become the
  // live form.
  function applyExtractData(data, label) {
    if (data.error) {
      setExtractStatus("Error: " + data.error);
      return;
    }
    extractId = data.extractId;
    fields = data.fields || [];
    currentWorkflowLabel = label;
    setExtractStatus(fields.length + " editable field(s) found.");
    renderFields();
  }

  function extractWorkflow(file) {
    if (!file) return;
    setExtractStatus("Reading workflow…");
    els.fieldsWrap.innerHTML = "";
    els.actions.style.display = "none";

    var form = new FormData();
    form.append("png", file);

    fetch("/api/ng/comfy/extract", { method: "POST", body: form })
      .then(function (r) { return r.json(); })
      .then(function (data) { applyExtractData(data, file.name); })
      .catch(function (e) {
        setExtractStatus("Error: " + e.message);
      });
  }

  // Lets a past output (or input) image double as its own workflow PNG
  // -- reads it straight off disk server-side (routes/comfyNG.py's
  // extract_comfy_server_image_ng), no download/upload round trip.
  function loadServerImageAsWorkflow(name, type) {
    var parts = String(name).split("/");
    var filename = parts.pop();
    var subfolder = parts.join("/");
    setExtractStatus("Reading workflow…");
    els.fieldsWrap.innerHTML = "";
    els.actions.style.display = "none";

    var url = "/api/ng/comfy/server-images/extract?filename=" + encodeURIComponent(filename) +
      "&subfolder=" + encodeURIComponent(subfolder) + "&type=" + type;
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) { applyExtractData(data, name); })
      .catch(function (e) {
        setExtractStatus("Error: " + e.message);
      });
  }

  // Arms/disarms a field as the target for the next click in the server-
  // images gallery. Only one field can be armed at a time -- arming a
  // second clears the first's highlight, same single-target model as the
  // gallery click itself (see pickServerImage).
  function setActiveImageField(f, wrapEl) {
    if (activeImageWrapEl) activeImageWrapEl.classList.remove("ng-comfy-image-field-target");
    activeImageField = f;
    activeImageWrapEl = f ? wrapEl : null;
    if (activeImageWrapEl) activeImageWrapEl.classList.add("ng-comfy-image-field-target");
  }

  // The "ComfyUI's own input images" gallery in the left rail -- images
  // that already exist on whatever machine ComfyUI runs on (see
  // routes/comfyNG.py's list_comfy_server_images_ng), for the common
  // multi-reference case (identity ref / scene ref / clothing ref, say)
  // where none of those source files are on *this* device to paste or
  // pick from disk. Loaded once at init, same as the saved-workflow
  // library -- the input folder isn't scoped to whichever workflow is
  // currently loaded, so there's nothing to re-fetch on Generate/extract.
  var serverImagesType = "input"; // "input" | "output", see setServerImagesType
  var serverImagesAll = []; // full filename list for serverImagesType, fetched once
  var serverImagesFilter = ""; // lowercased substring, see getFilteredServerImages
  var serverImagesPage = 0; // 0-indexed, into the filtered list
  var SERVER_IMAGES_PAGE_SIZE = 10;

  // Filtering happens entirely against the in-memory serverImagesAll --
  // no re-fetch, same reasoning as paging itself (see loadServerImages).
  function getFilteredServerImages() {
    if (!serverImagesFilter) return serverImagesAll;
    return serverImagesAll.filter(function (name) {
      return name.toLowerCase().indexOf(serverImagesFilter) !== -1;
    });
  }

  function loadServerImages() {
    if (!els.serverImages) return;
    els.serverImages.innerHTML = "";
    if (els.serverImagesPager) els.serverImagesPager.style.display = "none";
    var loading = document.createElement("p");
    loading.className = "ng-placeholder";
    loading.textContent = "Loading…";
    els.serverImages.appendChild(loading);

    fetch("/api/ng/comfy/server-images?type=" + serverImagesType)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          els.serverImages.innerHTML = "";
          var p = document.createElement("p");
          p.className = "ng-placeholder";
          p.textContent = "Couldn't list ComfyUI's " + serverImagesType + " images: " + data.error;
          els.serverImages.appendChild(p);
          return;
        }
        serverImagesAll = data.images || [];
        serverImagesPage = 0;
        renderServerImagesPage();
      })
      .catch(function (e) {
        els.serverImages.innerHTML = "";
        var p = document.createElement("p");
        p.className = "ng-placeholder";
        p.textContent = "Couldn't list ComfyUI's " + serverImagesType + " images: " + e.message;
        els.serverImages.appendChild(p);
      });
  }

  // Switches the gallery between ComfyUI's input/ and output/ folders
  // (routes/comfyNG.py's list_comfy_server_images_ng) -- input/ for
  // stuff dropped there directly, output/ for stuff ComfyUI itself
  // generated (previous renders you want to feed back in as a ref).
  function setServerImagesType(type) {
    if (type === serverImagesType) return;
    serverImagesType = type;
    if (els.imagesTabInput) els.imagesTabInput.classList.toggle("active", type === "input");
    if (els.imagesTabOutput) els.imagesTabOutput.classList.toggle("active", type === "output");
    loadServerImages();
  }

  // The filename list itself (serverImagesAll) is cheap even at ~1000
  // entries -- one JSON array. What isn't cheap is a real <img> per
  // entry: each one is a live fetch through /api/ng/comfy/view, and an
  // iPad falls over trying to load/decode a thousand of those at once.
  // So the network list is fetched whole, but only rendered a page
  // (SERVER_IMAGES_PAGE_SIZE) at a time -- paging just re-slices the
  // array already in memory, no re-fetch.
  function renderServerImagesPage() {
    var filtered = getFilteredServerImages();
    var start = serverImagesPage * SERVER_IMAGES_PAGE_SIZE;
    var slice = filtered.slice(start, start + SERVER_IMAGES_PAGE_SIZE);
    renderServerImages(slice);

    if (els.serverImagesPager) {
      els.serverImagesPager.style.display = filtered.length > SERVER_IMAGES_PAGE_SIZE ? "flex" : "none";
    }
    if (els.serverImagesPrev) els.serverImagesPrev.disabled = serverImagesPage === 0;
    if (els.serverImagesNext) {
      els.serverImagesNext.disabled = start + SERVER_IMAGES_PAGE_SIZE >= filtered.length;
    }
    if (els.serverImagesPageLabel) {
      var totalPages = Math.max(1, Math.ceil(filtered.length / SERVER_IMAGES_PAGE_SIZE));
      els.serverImagesPageLabel.textContent = (serverImagesPage + 1) + " / " + totalPages;
    }
  }

  // Reuses the saved-workflow card look (.ng-comfy-workflow-list/-card,
  // see styleNG.css) -- same "grid of square thumbnails with a filename
  // caption" shape, just images instead of saved workflows. `images` is
  // already just the current page's slice -- see renderServerImagesPage.
  function renderServerImages(images) {
    els.serverImages.innerHTML = "";
    if (!images.length) {
      var p = document.createElement("p");
      p.className = "ng-placeholder";
      p.textContent = serverImagesFilter
        ? "No images match \"" + serverImagesFilter + "\"."
        : "No images in ComfyUI's " + serverImagesType + " folder.";
      els.serverImages.appendChild(p);
      return;
    }
    var type = serverImagesType;
    images.forEach(function (name) {
      var parts = String(name).split("/");
      var filename = parts.pop();
      var subfolder = parts.join("/");

      var card = document.createElement("div");
      card.className = "ng-comfy-workflow-card";
      card.title = "Use " + name;

      var img = document.createElement("img");
      img.src = "/api/ng/comfy/view?filename=" + encodeURIComponent(filename) +
        "&subfolder=" + encodeURIComponent(subfolder) + "&type=" + type;
      img.alt = "";
      card.appendChild(img);

      var label = document.createElement("div");
      label.className = "ng-comfy-workflow-name";
      label.textContent = name;
      card.appendChild(label);

      var loadWfBtn = document.createElement("button");
      loadWfBtn.type = "button";
      loadWfBtn.className = "ng-comfy-server-image-extract";
      loadWfBtn.textContent = "Load as workflow";
      loadWfBtn.title = "Read this image's own embedded ComfyUI workflow instead of using it as a reference";
      loadWfBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        loadServerImageAsWorkflow(name, type);
      });
      card.appendChild(loadWfBtn);

      card.addEventListener("click", function () { pickServerImage(name, type); });
      els.serverImages.appendChild(card);
    });
  }

  function pickServerImage(filename, type) {
    if (!activeImageField) {
      alert('Click "Pick from server list" on an image field first, then choose one here.');
      return;
    }
    // Anything other than input/ needs ComfyUI's own "name [output]"
    // annotation (folder_paths.get_annotated_filepath) so a LoadImage
    // node resolves it from the right folder at execution time --
    // input/ files stay a bare filename, same as ComfyUI's own widget
    // would set.
    var value = type === "output" ? filename + " [output]" : filename;
    activeImageField._applyServerFilename(value);
    setActiveImageField(null, null);
  }

  // Saved-workflow library -- same {extractId, fields} result as picking
  // a PNG by hand, just sourced from routes/comfyNG.py's own saved copy
  // instead of an upload. See that file's save/load routes for why: the
  // file-picker/paste flow above needs access to whatever machine's disk
  // the PNG lives on, which the iPad doesn't have for this Mac.
  function loadWorkflowList() {
    if (!els.workflowList) return;
    fetch("/api/ng/comfy/workflows")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderWorkflowList(data.workflows || []);
      })
      .catch(function () {
        // Quiet failure -- the file-picker/paste flow still works even
        // if the library list can't load, so don't block on it.
      });
  }

  function renderWorkflowList(workflows) {
    els.workflowList.innerHTML = "";
    if (!workflows.length) {
      var p = document.createElement("p");
      p.className = "ng-placeholder";
      p.textContent = "No saved workflows yet.";
      els.workflowList.appendChild(p);
      return;
    }
    workflows.forEach(function (wf) {
      var card = document.createElement("div");
      card.className = "ng-comfy-workflow-card";
      card.title = "Load " + wf.name;

      var img = document.createElement("img");
      img.src = "/api/ng/comfy/workflows/" + wf.id + "/thumb";
      img.alt = "";
      card.appendChild(img);

      var name = document.createElement("div");
      name.className = "ng-comfy-workflow-name";
      name.textContent = wf.name;
      card.appendChild(name);

      var del = document.createElement("button");
      del.type = "button";
      del.className = "ng-comfy-workflow-delete";
      del.textContent = "×";
      del.title = "Delete " + wf.name;
      del.addEventListener("click", function (e) {
        e.stopPropagation();
        if (!confirm('Delete saved workflow "' + wf.name + '"?')) return;
        fetch("/api/ng/comfy/workflows/" + wf.id, { method: "DELETE" })
          .then(function () { loadWorkflowList(); });
      });
      card.appendChild(del);

      card.addEventListener("click", function () {
        loadSavedWorkflow(wf.id, wf.name);
      });

      els.workflowList.appendChild(card);
    });
  }

  function loadSavedWorkflow(wfId, name) {
    setExtractStatus("Loading saved workflow…");
    els.fieldsWrap.innerHTML = "";
    els.actions.style.display = "none";

    fetch("/api/ng/comfy/workflows/" + wfId + "/load", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          setExtractStatus("Error: " + data.error);
          return;
        }
        extractId = data.extractId;
        fields = data.fields || [];
        currentWorkflowLabel = name;
        setExtractStatus(fields.length + " editable field(s) found.");
        renderFields();
      })
      .catch(function (e) {
        setExtractStatus("Error: " + e.message);
      });
  }

  function handlePngPaste(e) {
    var items = (e.clipboardData && e.clipboardData.items) || [];
    for (var i = 0; i < items.length; i++) {
      if (items[i].kind === "file" && /^image\//.test(items[i].type)) {
        e.preventDefault();
        extractWorkflow(items[i].getAsFile());
        return;
      }
    }
  }

  // A frozen copy of the live fields at the moment Generate is clicked --
  // queuing means a second click can happen (and its own upload/submit
  // round trip can finish) before an earlier click's does, and both read
  // this snapshot rather than the shared, still-being-edited `fields`
  // array, or two queued jobs could end up using each other's values.
  // File objects themselves are immutable, so sharing the same _pendingFile
  // reference between the live field and this snapshot is safe -- only
  // the pointer needs copying, not the bytes.
  function snapshotFields() {
    return fields.map(function (f) {
      return {
        nodeId: f.nodeId,
        inputName: f.inputName,
        valueType: f.valueType,
        value: f.value,
        pendingFile: f._pendingFile || null,
      };
    });
  }

  // Any image field with a picked/pasted replacement needs uploading to
  // ComfyUI first (its value has to become a filename on ComfyUI's own
  // server before the graph can reference it) -- everything else's
  // value is already exactly what gets sent. Mutates the snapshot in
  // place, not the live fields (those keep showing the pending image so
  // it can be reused for another queued variation without re-pasting).
  function uploadSnapshotImages(snap) {
    var uploads = snap
      .filter(function (f) { return f.valueType === "image" && f.pendingFile; })
      .map(function (f) {
        var form = new FormData();
        form.append("image", f.pendingFile);
        return fetch("/api/ng/comfy/upload-image", { method: "POST", body: form })
          .then(function (r) { return r.json(); })
          .then(function (data) {
            if (data.error) throw new Error(f.inputName + ": " + data.error);
            f.value = data.value;
          });
      });
    return Promise.all(uploads);
  }

  // ComfyUI already queues multiple /prompt submissions itself, and
  // every comfyNG.py route is fully job_id-scoped -- the only thing
  // stopping more than one in-flight job before this was the frontend
  // serializing everything behind a single currentJobId/busy flag.
  // Each Generate click now just pushes its own queue row instead.
  function startGenerate() {
    if (!extractId) return;
    var extractIdAtSubmit = extractId; // also snapshotted -- switching workflows mid-flight shouldn't retarget an in-flight submission
    var snap = snapshotFields();

    var item = {
      localId: "c" + (++localSeq),
      jobId: null,
      status: "submitting",
      label: currentWorkflowLabel,
      error: null,
      resultUrl: null,
      savedName: null,
    };
    queue.unshift(item);
    renderQueue();

    uploadSnapshotImages(snap)
      .then(function () {
        var overrides = {};
        snap.forEach(function (f) {
          overrides[f.nodeId + "." + f.inputName] = f.value;
        });
        return fetch("/api/ng/comfy/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ extractId: extractIdAtSubmit, overrides: overrides }),
        });
      })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        if (data.error) {
          item.status = "error";
          item.error = data.error;
          renderQueue();
          return;
        }
        item.jobId = data.jobId;
        item.status = "running";
        renderQueue();
        ensurePolling();
      })
      .catch(function (e) {
        item.status = "error";
        item.error = e.message;
        renderQueue();
      });
  }

  function ensurePolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollOnce, POLL_MS);
    pollOnce();
  }

  function pollOnce() {
    var active = queue.filter(function (q) { return q.jobId && q.status === "running"; });
    if (!active.length) {
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }
    active.forEach(function (item) {
      fetch("/api/ng/comfy/status/" + item.jobId)
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.error || data.status === "error") {
            item.status = "error";
            item.error = data.error || "unknown error";
            renderQueue();
            return;
          }
          if (data.status === "running") return; // no change, next tick
          item.status = "done";
          item.resultUrl = "/api/ng/comfy/result/" + item.jobId + "?t=" + Date.now();
          renderQueue();
        })
        .catch(function () {
          /* transient -- try again next tick */
        });
    });
  }

  function removeQueueItem(localId) {
    queue = queue.filter(function (q) { return q.localId !== localId; });
    renderQueue();
    // No cancel-in-flight endpoint on the ComfyUI routes today -- the row
    // just stops being shown/polled here, same best-effort-only removal
    // as Animate's queue for a job that's already done/failed anyway.
  }

  function saveWorkflowFromItem(item, btn) {
    var name = prompt("Name for this saved workflow:");
    if (!name || !name.trim()) return;

    btn.disabled = true;
    fetch("/api/ng/comfy/workflows", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobId: item.jobId, name: name.trim() }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          alert("Couldn't save: " + data.error);
          btn.disabled = false;
          return;
        }
        item.savedName = data.name;
        loadWorkflowList();
        renderQueue();
      })
      .catch(function (e) {
        alert("Couldn't save: " + e.message);
        btn.disabled = false;
      });
  }

  // Same row shape as Animate's queue (videogenNG.js's buildQueueRow) --
  // .ng-gen-queue-row/.meta/.status-row/.st/.pose-remove are shared,
  // generic classes already styled for exactly this. Reuses its
  // st-done/st-failed/st-rendering color classes too, just mapped from
  // this simpler 4-state model instead of duplicating that CSS.
  function buildQueueRow(item) {
    var row = document.createElement("div");
    var stClass = item.status === "error" ? "failed" : item.status === "done" ? "done" : "rendering";
    row.className = "ng-gen-queue-row st-" + stClass;

    row.appendChild(document.createElement("div")); // empty 42px thumbs column, keeps the shared grid

    var meta = document.createElement("div");
    meta.className = "meta";
    var title = document.createElement("div");
    title.className = "title";
    title.textContent = item.label || "ComfyUI render";
    meta.appendChild(title);
    row.appendChild(meta);

    var statusRow = document.createElement("div");
    statusRow.className = "status-row";
    var st = document.createElement("span");
    st.className = "st";
    st.textContent =
      item.status === "submitting" ? "starting…" :
      item.status === "running" ? "generating…" :
      item.status === "error" ? "failed" : "done";
    statusRow.appendChild(st);

    var rm = document.createElement("button");
    rm.type = "button";
    rm.className = "pose-remove";
    rm.title = "remove from queue";
    rm.textContent = "✕";
    rm.addEventListener("click", function () { removeQueueItem(item.localId); });
    statusRow.appendChild(rm);
    row.appendChild(statusRow);

    if (item.status === "error" && item.error) {
      var err = document.createElement("div");
      err.className = "log";
      err.style.color = "var(--ng-danger)";
      err.textContent = item.error;
      row.appendChild(err);
    } else if (item.status === "done" && item.resultUrl) {
      var img = document.createElement("img");
      img.src = item.resultUrl;
      img.alt = "";
      img.style.gridColumn = "1 / -1";
      img.style.width = "100%";
      img.style.borderRadius = "4px";
      img.style.marginTop = "4px";
      row.appendChild(img);

      var saveBtn = document.createElement("button");
      saveBtn.type = "button";
      saveBtn.className = "ng-gen-btn ng-gen-btn-quiet";
      saveBtn.style.gridColumn = "1 / -1";
      saveBtn.style.marginTop = "4px";
      if (item.savedName) {
        saveBtn.textContent = 'Saved as "' + item.savedName + '"';
        saveBtn.disabled = true;
      } else {
        saveBtn.textContent = "Save this workflow…";
        saveBtn.addEventListener("click", function () { saveWorkflowFromItem(item, saveBtn); });
      }
      row.appendChild(saveBtn);
    }

    return row;
  }

  function renderQueue() {
    if (!els.queue) return;
    if (els.queueCount) els.queueCount.textContent = queue.length ? "(" + queue.length + ")" : "";
    if (els.queueEmpty) els.queueEmpty.style.display = queue.length ? "none" : "";
    els.queue.innerHTML = "";
    queue.forEach(function (item) {
      els.queue.appendChild(buildQueueRow(item));
    });
  }

  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane || !els.main) return;
    inited = true;

    els.pngFileBtn.addEventListener("click", function () {
      els.pngFile.click();
    });
    els.pngFile.addEventListener("change", function () {
      if (els.pngFile.files && els.pngFile.files[0]) extractWorkflow(els.pngFile.files[0]);
      els.pngFile.value = "";
    });
    els.pngPaste.addEventListener("paste", handlePngPaste);
    els.generateBtn.addEventListener("click", startGenerate);
    if (els.imagesTabInput) els.imagesTabInput.addEventListener("click", function () { setServerImagesType("input"); });
    if (els.imagesTabOutput) els.imagesTabOutput.addEventListener("click", function () { setServerImagesType("output"); });
    if (els.serverImagesPrev) {
      els.serverImagesPrev.addEventListener("click", function () {
        if (serverImagesPage === 0) return;
        serverImagesPage -= 1;
        renderServerImagesPage();
      });
    }
    if (els.serverImagesNext) {
      els.serverImagesNext.addEventListener("click", function () {
        if ((serverImagesPage + 1) * SERVER_IMAGES_PAGE_SIZE >= getFilteredServerImages().length) return;
        serverImagesPage += 1;
        renderServerImagesPage();
      });
    }
    if (els.serverImagesFilter) {
      els.serverImagesFilter.addEventListener("input", function () {
        serverImagesFilter = els.serverImagesFilter.value.trim().toLowerCase();
        serverImagesPage = 0;
        renderServerImagesPage();
      });
    }
    loadWorkflowList();
    loadServerImages();
    renderQueue();
  }

  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "comfy");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    // Same "runs last, unconditional hide-when-on wins" deal as
    // HdMultiNG.sync -- see that file's own comment.
    if (on && els.controlsPane) els.controlsPane.style.display = "none";
    if (!on) return;
    init();
  }

  window.ComfyNG = { sync: sync };
})();
