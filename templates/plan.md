Yeah. Now that we've actually mapped the beast, I'd split it based on responsibility and dependency direction, not simply recreate the old four files.
The key is: CharacterProject and ProjectManager should survive as the core, while feature implementations move out around them.
I'd aim for something like this:
static/
├── appNG.js ← small bootstrap/orchestrator
│
├── ng-project.js ← CharacterProject
├── ng-project-manager.js ← ProjectManager
│
├── ng-render.js ← rendering the shared shell
├── ng-task-ui.js ← footer/tasks
│
├── ng-video.js ← video ingestion/playback/analysis
├── ng-immich.js ← Immich ingestion/search/analysis
├── ng-folder.js ← folder/zip ingestion
│
├── ng-ring.js ← ring construction/rendering
├── ng-selection.js ← selection/export/selection UI
├── ng-pose.js ← pose picker/sorting
│
└── ng-generate.js ← character-sheet generation
But I'd actually do the extraction in a particular order, because otherwise we risk breaking the thing.

1. First: extract CharacterProject
   This is the heart of the new architecture.
   ng-project.js
   Everything that defines:
   new CharacterProject(...)
   and its state belongs there.
   Video state, Immich state, ring state, selections, job state, etc.
   The important rule becomes:
   CharacterProject owns state. It doesn't own UI rendering.

That gives us a really useful boundary. 2. Extract ProjectManager
Then:
ng-project-manager.js
Things like:
projects
activeId
getActive()
setTask()
switchProject()
createProject()
deleteProject()
render()
Eventually ProjectManager should mostly be orchestration.
Not:
// 600 lines of actual ring rendering
but:
project.task = "video";
this.render(); 3. Extract rendering
Then I'd tackle:
ng-render.js
This is probably one of the biggest wins.
Move:
renderTabs()
renderBottomBar()
renderLeftRail()
renderMain()
renderStage()
and the associated DOM rendering helpers.
Then the architecture starts becoming obvious:
ProjectManager
│
↓
render
│
└── ng-render.js 4. Split input sources
Then follow the old application's useful boundaries:
ng-video.js
ng-immich.js
ng-folder.js
This is where I'd not blindly reproduce the old files.
For example, video-specific things belong in ng-video.js, but they all receive/use the active CharacterProject.
So instead of hidden globals:
lastVideoRingState
we eventually have:
project.video
project.ring
project.selection
That's a much better architecture. 5. Ring and selection
I'd then separate the stuff that is currently probably tangled together:
ng-ring.js
ng-selection.js
ng-pose.js
Because these are conceptually different operations:
analysis results
↓
ring
↓
selection
↓
pose analysis / pose view
And this is where the old viz-render.js and selection-ui.js boundaries become useful reference points. 6. Generate last
Then:
ng-generate.js
The old phosphene-sheet.js gives us the behavioural specification.
But by the time we get there, Generate can simply say:
active CharacterProject
↓
ng-generate.js
↓
generation job
↓
project.generation
rather than having its own pile of global state.
And appNG.js?
This is the important bit.
It should end up boring.
Something approximately like:
appNG.js

initialise DOM
↓
initialise ProjectManager
↓
wire modules
↓
create/restore projects
↓
initial render
Ideally, six months from now you open appNG.js and think:
“Oh, that's the entry point.”

rather than:
“Jesus Christ, what's happening in here?” 😂

One other thing I'd deliberately avoid
I wouldn't create 15 tiny files.
That's just replacing one problem with another.
The old four-file split was useful because each file represented a meaningful chunk. I'd aim for perhaps 8–10 reasonably substantial modules, with appNG.js being very small.
And the extraction should be mechanical first, architectural second:
Move existing code → make it work → test → then improve the interfaces.

That way we're not simultaneously refactoring the architecture and changing behaviour.
Given what we've learned tonight, that's probably the safest route to turning the monolith back into something an AI can actually digest.
