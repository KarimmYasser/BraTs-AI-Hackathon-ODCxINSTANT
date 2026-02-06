/**
 * Brain Tumor Segmentation - Frontend Application
 */

class BrainTumorApp {
  constructor() {
    const configuredBase = document.body?.dataset?.apiBaseUrl || "";
    const isLocalStatic =
      window.location.hostname === "127.0.0.1" &&
      window.location.port === "3000";
    this.apiBaseUrl =
      configuredBase || (isLocalStatic ? "http://127.0.0.1:8080" : "");
    this.sessionId = null;
    this.selectedModel = "unet";
    this.currentSlice = 64;
    this.numSlices = 128;
    this.showOverlay = true;
    this.viewMode = "2d"; // '2d' or '3d'
    this.files = {
      t1: null,
      t1ce: null,
      t2: null,
      flair: null,
    };
    // Multi-view state
    this.multiviewSlices = {
      axial: 64,
      coronal: 64,
      sagittal: 64,
    };
    this.multiviewDimensions = null;
    this.multiviewModality = "flair";

    this.init();
  }

  init() {
    this.bindElements();
    this.bindEvents();
    this.loadDevices();
  }

  bindElements() {
    // Sections
    this.uploadSection = document.getElementById("uploadSection");
    this.modelSection = document.getElementById("modelSection");
    this.resultsSection = document.getElementById("resultsSection");

    // Upload elements
    this.uploadCards = document.querySelectorAll(".upload-card");
    this.uploadBtn = document.getElementById("uploadBtn");

    // Model elements
    this.modelCards = document.querySelectorAll(".model-card");
    this.segmentBtn = document.getElementById("segmentBtn");

    // Results elements
    this.sliceSlider = document.getElementById("sliceSlider");
    this.sliceNumber = document.getElementById("sliceNumber");
    this.totalSlices = document.getElementById("totalSlices");
    this.overlayToggle = document.getElementById("overlayToggle");
    this.prevSliceBtn = document.getElementById("prevSlice");
    this.nextSliceBtn = document.getElementById("nextSlice");
    this.newScanBtn = document.getElementById("newScanBtn");

    // View mode elements
    this.btn2DView = document.getElementById("btn2DView");
    this.btn3DView = document.getElementById("btn3DView");
    this.sliceViewer2D = document.getElementById("sliceViewer2D");
    this.multiviewDisplay = document.getElementById("multiviewDisplay");

    // Multi-view elements
    this.multiviewModality = document.getElementById("multiviewModality");
    this.axialSlider = document.getElementById("axialSlider");
    this.coronalSlider = document.getElementById("coronalSlider");
    this.sagittalSlider = document.getElementById("sagittalSlider");

    // Loading overlay
    this.loadingOverlay = document.getElementById("loadingOverlay");
    this.loadingText = document.getElementById("loadingText");

    // Device selector
    this.deviceSelect = document.getElementById("deviceSelect");
    this.deviceStatus = document.getElementById("deviceStatus");
  }

  bindEvents() {
    // File upload
    this.uploadCards.forEach((card) => {
      card.addEventListener("click", () => this.triggerFileInput(card));
      card.addEventListener("dragover", (e) => this.handleDragOver(e, card));
      card.addEventListener("dragleave", (e) => this.handleDragLeave(e, card));
      card.addEventListener("drop", (e) => this.handleDrop(e, card));
    });

    // File inputs
    ["t1", "t1ce", "t2", "flair"].forEach((mod) => {
      document.getElementById(`file-${mod}`).addEventListener("change", (e) => {
        this.handleFileSelect(mod, e.target.files[0]);
      });
    });

    // Upload button
    this.uploadBtn.addEventListener("click", () => this.uploadFiles());

    // Model selection
    this.modelCards.forEach((card) => {
      card.addEventListener("click", () =>
        this.selectModel(card.dataset.model),
      );
    });

    // Segment button
    this.segmentBtn.addEventListener("click", () => this.runSegmentation());

    // Slice navigation
    this.sliceSlider.addEventListener("input", (e) => {
      this.currentSlice = parseInt(e.target.value);
      this.updateSliceDisplay();
    });

    this.prevSliceBtn.addEventListener("click", () => {
      if (this.currentSlice > 0) {
        this.currentSlice--;
        this.sliceSlider.value = this.currentSlice;
        this.updateSliceDisplay();
      }
    });

    this.nextSliceBtn.addEventListener("click", () => {
      if (this.currentSlice < this.numSlices - 1) {
        this.currentSlice++;
        this.sliceSlider.value = this.currentSlice;
        this.updateSliceDisplay();
      }
    });

    // Overlay toggle
    this.overlayToggle.addEventListener("change", (e) => {
      this.showOverlay = e.target.checked;
      if (this.viewMode === "2d") {
        this.loadSlice(this.currentSlice);
      } else {
        this.loadMultiview();
      }
    });

    // View mode toggle
    this.btn2DView.addEventListener("click", () => this.setViewMode("2d"));
    this.btn3DView.addEventListener("click", () => this.setViewMode("3d"));

    // Multi-view controls
    if (this.multiviewModality) {
      this.multiviewModality.addEventListener("change", (e) => {
        this.multiviewModalityName = e.target.value;
        this.loadMultiview();
      });
    }

    if (this.axialSlider) {
      this.axialSlider.addEventListener("input", (e) => {
        this.multiviewSlices.axial = parseInt(e.target.value);
        document.getElementById("axialSliceNum").textContent = e.target.value;
        this.loadMultiview();
      });
    }

    if (this.coronalSlider) {
      this.coronalSlider.addEventListener("input", (e) => {
        this.multiviewSlices.coronal = parseInt(e.target.value);
        document.getElementById("coronalSliceNum").textContent = e.target.value;
        this.loadMultiview();
      });
    }

    if (this.sagittalSlider) {
      this.sagittalSlider.addEventListener("input", (e) => {
        this.multiviewSlices.sagittal = parseInt(e.target.value);
        document.getElementById("sagittalSliceNum").textContent =
          e.target.value;
        this.loadMultiview();
      });
    }

    // New scan button
    this.newScanBtn.addEventListener("click", () => this.resetApp());

    // Keyboard navigation
    document.addEventListener("keydown", (e) => {
      if (this.resultsSection.classList.contains("hidden")) return;

      if (e.key === "ArrowLeft") {
        this.prevSliceBtn.click();
      } else if (e.key === "ArrowRight") {
        this.nextSliceBtn.click();
      }
    });
  }

  async loadDevices() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/devices`);
      const data = await response.json();

      // Update device selector
      if (this.deviceSelect) {
        // Clear existing options
        this.deviceSelect.innerHTML = "";

        data.devices.forEach((device) => {
          const option = document.createElement("option");
          option.value = device.id;
          option.textContent = device.name;
          option.disabled = !device.available;
          this.deviceSelect.appendChild(option);
        });

        // Set current device
        const currentDevice = data.current_device
          .replace("cuda:", "cuda")
          .replace("cpu:", "cpu");
        if (currentDevice.includes("cuda")) {
          this.deviceSelect.value = "cuda";
        } else {
          this.deviceSelect.value = "cpu";
        }

        // Show status
        this.updateDeviceStatus(data.current_device);

        // Bind change event
        this.deviceSelect.addEventListener("change", (e) =>
          this.changeDevice(e.target.value),
        );
      }
    } catch (error) {
      if (this.deviceStatus) {
        this.deviceStatus.textContent = "Connecting...";
      }
    }
  }

  async changeDevice(deviceId) {
    try {
      const formData = new FormData();
      formData.append("device_name", deviceId);

      const response = await fetch(`${this.apiBaseUrl}/device`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to change device");
      }

      const data = await response.json();
      this.updateDeviceStatus(data.device);
    } catch (error) {
      alert("Error changing device: " + error.message);
      // Reload devices to reset selector
      this.loadDevices();
    }
  }

  updateDeviceStatus(device) {
    if (this.deviceStatus) {
      const icon = device.includes("cuda") ? "⚡" : "💻";
      this.deviceStatus.textContent = icon;
      this.deviceStatus.title = `Current device: ${device}`;
    }
  }

  triggerFileInput(card) {
    const modality = card.dataset.modality;
    document.getElementById(`file-${modality}`).click();
  }

  handleDragOver(e, card) {
    e.preventDefault();
    card.classList.add("active");
  }

  handleDragLeave(e, card) {
    e.preventDefault();
    card.classList.remove("active");
  }

  handleDrop(e, card) {
    e.preventDefault();
    card.classList.remove("active");

    const modality = card.dataset.modality;
    const file = e.dataTransfer.files[0];

    if (file && (file.name.endsWith(".nii") || file.name.endsWith(".nii.gz"))) {
      this.handleFileSelect(modality, file);
    }
  }

  handleFileSelect(modality, file) {
    if (!file) return;

    this.files[modality] = file;

    // Update UI
    const card = document.querySelector(`[data-modality="${modality}"]`);
    const filename = document.getElementById(`filename-${modality}`);
    const status = document.getElementById(`status-${modality}`);

    card.classList.add("active");
    filename.textContent = file.name;
    status.textContent = "✓";
    status.style.color = "#22c55e";

    // Check if all files are selected
    this.checkAllFilesSelected();
  }

  checkAllFilesSelected() {
    const allSelected = Object.values(this.files).every((f) => f !== null);
    this.uploadBtn.disabled = !allSelected;
  }

  async uploadFiles() {
    this.showLoading("Uploading and processing files...");

    try {
      const formData = new FormData();
      formData.append("t1", this.files.t1);
      formData.append("t1ce", this.files.t1ce);
      formData.append("t2", this.files.t2);
      formData.append("flair", this.files.flair);

      const response = await fetch(`${this.apiBaseUrl}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = "Upload failed";
        try {
          const error = await response.json();
          message = error.detail || message;
        } catch (err) {
          const text = await response.text();
          if (text) message = text;
        }
        throw new Error(message);
      }

      const data = await response.json();
      this.sessionId = data.session_id;
      this.numSlices = data.num_slices;
      this.currentSlice = Math.floor(this.numSlices / 2);

      // Show model selection
      this.uploadSection.classList.add("hidden");
      this.modelSection.classList.remove("hidden");

      this.hideLoading();
    } catch (error) {
      this.hideLoading();
      alert("Error uploading files: " + error.message);
    }
  }

  selectModel(modelId) {
    this.selectedModel = modelId;

    this.modelCards.forEach((card) => {
      card.classList.toggle("selected", card.dataset.model === modelId);
    });
  }

  async runSegmentation() {
    this.showLoading("Running segmentation... This may take a moment.");

    try {
      const formData = new FormData();
      formData.append("model_name", this.selectedModel);

      const response = await fetch(
        `${this.apiBaseUrl}/segment/${this.sessionId}`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        let message = "Segmentation failed";
        try {
          const error = await response.json();
          message = error.detail || message;
        } catch (err) {
          const text = await response.text();
          if (text) message = text;
        }
        throw new Error(message);
      }

      const data = await response.json();

      // Update statistics
      document.getElementById("stat-ncr").textContent =
        data.tumor_statistics.ncr.toLocaleString();
      document.getElementById("stat-ed").textContent =
        data.tumor_statistics.ed.toLocaleString();
      document.getElementById("stat-et").textContent =
        data.tumor_statistics.et.toLocaleString();

      // Update slider
      this.sliceSlider.max = this.numSlices - 1;
      this.sliceSlider.value = this.currentSlice;
      this.totalSlices.textContent = this.numSlices;

      // Show results
      this.modelSection.classList.add("hidden");
      this.resultsSection.classList.remove("hidden");

      // Load initial slice
      await this.loadSlice(this.currentSlice);

      this.hideLoading();
    } catch (error) {
      this.hideLoading();
      alert("Error running segmentation: " + error.message);
    }
  }

  async loadSlice(sliceIdx) {
    try {
      const response = await fetch(
        `${this.apiBaseUrl}/slice/${this.sessionId}/${sliceIdx}?overlay=${this.showOverlay}`,
      );

      if (!response.ok) {
        throw new Error("Failed to load slice");
      }

      const data = await response.json();

      // Update images
      const suffix = this.showOverlay ? "" : "_no_overlay";

      ["t1", "t1ce", "t2", "flair"].forEach((mod) => {
        const img = document.getElementById(`img-${mod}`);
        const imageData = this.showOverlay
          ? data.images[mod]
          : data.images[`${mod}_no_overlay`];
        img.src = `data:image/png;base64,${imageData}`;
      });

      // Segmentation view
      if (data.images.segmentation) {
        document.getElementById("img-segmentation").src =
          `data:image/png;base64,${data.images.segmentation}`;
      }

      // Update slice number
      this.sliceNumber.textContent = sliceIdx;
    } catch (error) {
      console.error("Error loading slice:", error);
    }
  }

  updateSliceDisplay() {
    this.sliceNumber.textContent = this.currentSlice;
    this.loadSlice(this.currentSlice);
  }

  setViewMode(mode) {
    this.viewMode = mode;

    // Update button states
    this.btn2DView.classList.toggle("active", mode === "2d");
    this.btn3DView.classList.toggle("active", mode === "3d");

    // Toggle displays
    if (mode === "2d") {
      this.sliceViewer2D.classList.remove("hidden");
      this.multiviewDisplay.classList.add("hidden");
    } else {
      this.sliceViewer2D.classList.add("hidden");
      this.multiviewDisplay.classList.remove("hidden");
      this.loadMultiview();
    }
  }

  async loadMultiview() {
    if (!this.sessionId) return;

    try {
      const modality = this.multiviewModality?.value || "flair";
      const url =
        `${this.apiBaseUrl}/multiview/${this.sessionId}?` +
        `axial=${this.multiviewSlices.axial}&` +
        `coronal=${this.multiviewSlices.coronal}&` +
        `sagittal=${this.multiviewSlices.sagittal}&` +
        `modality=${modality}&` +
        `overlay=${this.showOverlay}`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error("Failed to load multi-view");
      }

      const data = await response.json();

      // Store dimensions for slider limits
      if (data.dimensions && !this.multiviewDimensions) {
        this.multiviewDimensions = data.dimensions;
        this.updateMultiviewSliders(data.dimensions);
      }

      // Update images
      const suffix = this.showOverlay ? "" : "_no_overlay";

      document.getElementById("img-axial").src =
        `data:image/png;base64,${this.showOverlay ? data.images.axial : data.images.axial_no_overlay}`;
      document.getElementById("img-coronal").src =
        `data:image/png;base64,${this.showOverlay ? data.images.coronal : data.images.coronal_no_overlay}`;
      document.getElementById("img-sagittal").src =
        `data:image/png;base64,${this.showOverlay ? data.images.sagittal : data.images.sagittal_no_overlay}`;
    } catch (error) {
      console.error("Error loading multi-view:", error);
    }
  }

  updateMultiviewSliders(dimensions) {
    // Update slider ranges based on volume dimensions
    if (this.axialSlider) {
      this.axialSlider.max = dimensions.width - 1;
      this.axialSlider.value = Math.floor(dimensions.width / 2);
      this.multiviewSlices.axial = parseInt(this.axialSlider.value);
      document.getElementById("axialSliceNum").textContent =
        this.axialSlider.value;
    }
    if (this.coronalSlider) {
      this.coronalSlider.max = dimensions.height - 1;
      this.coronalSlider.value = Math.floor(dimensions.height / 2);
      this.multiviewSlices.coronal = parseInt(this.coronalSlider.value);
      document.getElementById("coronalSliceNum").textContent =
        this.coronalSlider.value;
    }
    if (this.sagittalSlider) {
      this.sagittalSlider.max = dimensions.depth - 1;
      this.sagittalSlider.value = Math.floor(dimensions.depth / 2);
      this.multiviewSlices.sagittal = parseInt(this.sagittalSlider.value);
      document.getElementById("sagittalSliceNum").textContent =
        this.sagittalSlider.value;
    }
  }

  resetApp() {
    // Clear session
    if (this.sessionId) {
      fetch(`${this.apiBaseUrl}/session/${this.sessionId}`, {
        method: "DELETE",
      }).catch(() => {});
    }

    // Reset state
    this.sessionId = null;
    this.files = { t1: null, t1ce: null, t2: null, flair: null };
    this.currentSlice = 64;
    this.numSlices = 128;
    this.viewMode = "2d";
    this.multiviewSlices = { axial: 64, coronal: 64, sagittal: 64 };
    this.multiviewDimensions = null;

    // Reset UI
    this.uploadCards.forEach((card) => {
      const modality = card.dataset.modality;
      card.classList.remove("active");
      document.getElementById(`filename-${modality}`).textContent =
        "No file selected";
      document.getElementById(`status-${modality}`).textContent = "";
      document.getElementById(`file-${modality}`).value = "";
    });

    this.uploadBtn.disabled = true;

    // Reset view mode buttons
    this.btn2DView.classList.add("active");
    this.btn3DView.classList.remove("active");
    this.sliceViewer2D.classList.remove("hidden");
    this.multiviewDisplay.classList.add("hidden");

    // Show upload section
    this.resultsSection.classList.add("hidden");
    this.modelSection.classList.add("hidden");
    this.uploadSection.classList.remove("hidden");
  }

  showLoading(message) {
    this.loadingText.textContent = message;
    this.loadingOverlay.classList.remove("hidden");
  }

  hideLoading() {
    this.loadingOverlay.classList.add("hidden");
  }
}

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.app = new BrainTumorApp();
});
