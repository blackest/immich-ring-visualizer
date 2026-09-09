// domNG.js
// Shared DOM handles for the split NG runtime.
// This file must load after indexNG.html markup and before the feature modules.

var tabsEl = document.getElementById("ng-tabs");
var mainPlaceholderEl = document.getElementById("ng-main-placeholder");
var newProjectBtn = document.getElementById("ng-new-project");
var loadProjectBtn = document.getElementById("ng-load-project");
var taskButtons = Array.from(document.querySelectorAll(".ng-task-btn"));
var videoAudioEl = document.getElementById("ng-video-audio");

var leftRailEl = document.getElementById("ng-leftrail");
var leftRailEmptyEl = document.getElementById("ng-leftrail-empty");
var leftRailBodyEl = document.getElementById("ng-leftrail-body");
var collapseAllBtn = document.getElementById("ng-leftrail-collapse-all");
var resizeHandleEl = document.getElementById("ng-leftrail-resize-handle");
var controlsPaneEl = document.getElementById("ng-controls-pane");
var splitterEl = document.getElementById("ng-splitter");

var ringScaleInput = document.getElementById("ng-ring-scale-input");
var ringScaleVal = document.getElementById("ng-ring-scale-val");
var squeezeSlider = document.getElementById("ng-ring-squeeze-slider");
var squeezeVal = document.getElementById("ng-ring-squeeze-val");
var ringSortCbs = Array.from(document.querySelectorAll(".ng-ring-sort-cb"));
var sharpEnableCb = document.getElementById("ng-sharp-squeeze-enable");
var sharpControlsEl = document.getElementById("ng-sharp-squeeze-controls");
var sharpSlider = document.getElementById("ng-sharp-squeeze-slider");
var sharpVal = document.getElementById("ng-sharp-squeeze-val");
var findNeutralBtn = document.getElementById("ng-find-neutral-btn");
var neutralPoseReadoutEl = document.getElementById("ng-neutral-pose-readout");

var videoAnalysisBodyEl = document.getElementById("ng-video-analysis-body");
var analysisStartInput = document.getElementById("ng-analysis-start-sec");
var analysisEndInput = document.getElementById("ng-analysis-end-sec");
var simThresholdInput = document.getElementById("ng-sim-threshold");
var blurThresholdInput = document.getElementById("ng-blur-threshold");
var cacheFormatPngCb = document.getElementById("ng-cache-format-png");
var analysisStatusEl = document.getElementById("ng-analysis-status");
var loadFolderBtn = document.getElementById("ng-btn-load-folder");
var loadZipBtn = document.getElementById("ng-btn-load-zip");
var folderRefIndexInput = document.getElementById("ng-folder-ref-index");
var analysisSectionTitleEl = document.getElementById(
  "ng-analysis-section-title",
);
var folderRowEl = document.querySelector(".ng-folder-row");
var videoRangeRowEl = document.querySelector(".ng-video-range");

var previewHintEl = document.getElementById("ng-preview-hint");
var previewCanvasEl = document.getElementById("ng-preview-canvas");
var previewControlsScrollEl = document.getElementById(
  "ng-preview-controls-scroll",
);
var frameCounterEl = document.getElementById("ng-frame-counter");
var rewindBtn = document.getElementById("ng-btn-rewind-frame");
var prevFrameBtn = document.getElementById("ng-btn-prev-frame");
var playBtn = document.getElementById("ng-btn-play-frames");
var stopBtn = document.getElementById("ng-btn-stop-frames");
var nextFrameBtn = document.getElementById("ng-btn-next-frame");
var startAnalysisBtn = document.getElementById("ng-btn-start-analysis");
var popoutVideoBtn = document.getElementById("ng-btn-popout-video");

var playbackModalEl = document.getElementById("ng-playback-modal");
var playbackModalTitleEl = document.getElementById("ng-playback-modal-title");
var playbackModalCloseBtn = document.getElementById("ng-playback-modal-close");
var playbackVideoEl = document.getElementById("ng-playback-video");
var playbackPrevFrameBtn = document.getElementById("ng-playback-prev-frame");
var playbackNextFrameBtn = document.getElementById("ng-playback-next-frame");

var hoverPanel = document.getElementById("ng-preview-hover-panel");
var hoverImg = document.getElementById("ng-preview-hover-img");
var hoverCaption = document.getElementById("ng-preview-hover-caption");
var hoverTimer = null;

var stageWrapEl = document.getElementById("ng-stage-wrap");
var stageEl = document.getElementById("ng-stage");
var poseListViewEl = document.getElementById("ng-pose-list-view");
var poseListScrubberEl = document.getElementById("ng-pose-list-scrubber");
var poseScrubSliderEl = document.getElementById("ng-pose-scrub-slider");
var poseScrubLeftEl = document.getElementById("ng-pose-scrub-left");
var poseScrubRightEl = document.getElementById("ng-pose-scrub-right");
var hudModeEl = document.getElementById("ng-hud-mode");
var hudFilenameEl = document.getElementById("ng-hud-filename");
var sidebarEl = document.getElementById("ng-sidebar");
var toggleListBtn = document.getElementById("ng-toggle-list-btn");
var sidebarCurrentImgEl = document.getElementById("ng-sidebar-current-img");
var sidebarCurrentFnameEl = document.getElementById("ng-sidebar-current-fname");
var sidebarCurrentModeEl = document.getElementById("ng-sidebar-current-mode");
var sidebarCurrentDetailEl = document.getElementById(
  "ng-sidebar-current-detail",
);
var framesSectionEl = document.getElementById("ng-frames-section");
var framesSectionCountEl = document.getElementById("ng-frames-section-count");
var listBodyFramesEl = document.getElementById("ng-list-body-frames");
var framesSelectAllBtn = document.getElementById("ng-frames-select-all");
var framesDeselectAllBtn = document.getElementById("ng-frames-deselect-all");
var rankedSortRadios = Array.from(
  document.querySelectorAll(".ng-ranked-sort-cb"),
);

var immichSectionEl = document.getElementById("ng-immich-section");
var immichSectionCountEl = document.getElementById("ng-immich-section-count");
var listBodyImmichEl = document.getElementById("ng-list-body-immich");
var immichSelectAllBtn = document.getElementById("ng-immich-select-all");
var immichDeselectAllBtn = document.getElementById("ng-immich-deselect-all");
var immichRefreshBtn = document.getElementById("ng-immich-refresh-btn");
var immichSaveSelectedBtn = document.getElementById("ng-immich-save-selected");
var immichExportResultEl = document.getElementById("ng-immich-export-result");
var immichSelectionBarEl = document.getElementById("ng-immich-selection-bar");
var immichSelectionCountEl = document.querySelector(
  "#ng-immich-selection-count span",
);
var immichAnalyzeSelectedBtn = document.getElementById(
  "ng-immich-analyze-selected-btn",
);
var immichExportSelectedBtn = document.getElementById(
  "ng-immich-export-selected-btn",
);
var immichViewSelectedBtn = document.getElementById(
  "ng-immich-view-selected-btn",
);
var immichClearSelectedBtn = document.getElementById(
  "ng-immich-clear-selected-btn",
);

var immichSearchInput = document.getElementById("ng-immich-search-input");
var immichRandomFaceBtn = document.getElementById("ng-immich-random-face-btn");
var immichSearchStatusEl = document.getElementById("ng-immich-search-status");
var immichSearchResultsEl = document.getElementById("ng-immich-search-results");
var immichAnalyzeBtn = document.getElementById("ng-immich-analyze-btn");
var immichAnalyzeRefIndexInput = document.getElementById(
  "ng-immich-analyze-ref-index",
);
var immichSimThresholdInput = document.getElementById(
  "ng-immich-sim-threshold",
);
var immichBlurThresholdInput = document.getElementById(
  "ng-immich-blur-threshold",
);
var immichAnalysisStatusEl = document.getElementById(
  "ng-immich-analysis-status",
);
var framesSectionTitleEl = document.getElementById("ng-frames-section-title");

var videoAnalysisSectionEl = document.querySelector(
  '.panel-section[data-section="video-analysis"]',
);
var searchSectionEl = document.querySelector(
  '.panel-section[data-section="search"]',
);

var posePickerEmptyEl = document.getElementById("ng-pose-picker-empty");
var posePickerControlsEl = document.getElementById("ng-pose-picker-controls");
var posePickerPitchSlider = document.getElementById("ng-pose-picker-pitch");
var posePickerPitchNum = document.getElementById("ng-pose-picker-pitch-num");
var posePickerYawSlider = document.getElementById("ng-pose-picker-yaw");
var posePickerYawNum = document.getElementById("ng-pose-picker-yaw-num");
var posePickerToleranceEnable = document.getElementById(
  "ng-pose-picker-tolerance-enable",
);
var posePickerToleranceVal = document.getElementById(
  "ng-pose-picker-tolerance-val",
);
var posePickerGridEl = document.getElementById("ng-pose-picker-grid");
var posePickerSelectBtn = document.getElementById("ng-pose-picker-select-btn");
var posePickerCountEl = document.getElementById("ng-pose-picker-count");
var posePickerCellsNG = posePickerGridEl ? [] : [];

var scalePickerEmptyEl = document.getElementById("ng-scale-picker-empty");
var scalePickerControlsEl = document.getElementById("ng-scale-picker-controls");
var scalePickerSlider = document.getElementById("ng-scale-picker-slider");
var scalePickerNum = document.getElementById("ng-scale-picker-num");
var scalePickerToleranceEnable = document.getElementById(
  "ng-scale-picker-tolerance-enable",
);
var scalePickerToleranceVal = document.getElementById(
  "ng-scale-picker-tolerance-val",
);
var scalePickerGridEl = document.getElementById("ng-scale-picker-grid");
var scalePickerSelectBtn = document.getElementById(
  "ng-scale-picker-select-btn",
);
var scalePickerCountEl = document.getElementById("ng-scale-picker-count");
var scalePickerCellsNG = scalePickerGridEl ? [] : [];

var ngSelectedModal = document.getElementById("ng-selected-modal");
var selectedModalOverlay = ngSelectedModal;
var selectedModalCloseBtn = document.getElementById("ng-selected-modal-close");
