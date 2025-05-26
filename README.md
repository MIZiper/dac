# Data Action Context

DAC provides a minimal frame for (measurement) data analysis
if you want to:

- Visualize data, process and interact
- Customize your analysis
- Save the analysis and load back
- Enable multiple analysis of same processing (like batch analysis)
- Link different analysis

Example of DAC user interface as shown below:

![DAC GUI](./doc/dac-gui.png)

## Concepts

### Data & Action

At its core, DAC operates on two fundamental types of nodes: **Data nodes** and **Action nodes**.

*   **Data nodes** represent various forms of information within your analysis workflow. This can include raw measurement data (e.g., time series, sensor readings), processed data (e.g., frequency spectra, statistical summaries), or configuration parameters. Each Data node holds a specific piece of information and its associated metadata (like units, sampling rates).

*   **Action nodes** perform operations. They typically consume one or more Data nodes as input, along with user-defined parameters, and produce new Data nodes as output or generate visualizations.
    *   *Processing Actions*: These are often non-interactive and can be computationally intensive. They transform input data to produce new data. For example, an "FFT" Action might take a "TimeData" node and produce a "FrequencySpectrum" Data node.
    *   *Visualizing Actions*: These actions are primarily for displaying data graphically. They are often interactive, allowing for zooming, panning, or custom interactions (e.g., marking points on a plot). They generally do not produce new Data nodes in the main workflow but might have internal data representations for the plot.

The workflow in DAC is built by connecting Data nodes to Action nodes, forming a directed graph that represents the analysis pipeline.

### Interaction

Interaction with the DAC framework primarily occurs through its graphical user interface (GUI):

*   **Node Management**: Users can add, remove, and connect Data and Action nodes to build or modify their analysis workflow.
*   **Parameter Configuration**: Action nodes have configurable parameters that can be adjusted through a dedicated editor panel. For instance, an FFT Action might have parameters for window type, segment length, and overlap.
*   **Data Visualization**: Visualizing Actions render data in plot windows. These plots can be interactive, offering features like zoom/pan, and can also support custom event handling defined by the specific Action (e.g., clicking a point on a time series plot to trigger a calculation based on that point).
*   **Context Switching**: Users can switch between different Contexts to manage and compare various analyses.
*   **Auxiliary Tools**: Quick Tasks and Quick Actions (see below) provide further streamlined interaction points.

While DAC can be scripted and its components used in environments like Jupyter notebooks, its primary interaction model is GUI-driven for building and running analysis workflows.

### Context

A **Context** provides a way to manage and isolate different sets of data, parameters, and analysis variations within a single project. This is particularly useful when:

*   Analyzing multiple measurements taken under different conditions (e.g., "Machine_Test_A", "Machine_Test_B").
*   Comparing the results of the same analysis pipeline with slight modifications to parameters or input data.
*   Keeping related analyses organized and separate.

Each Context can have its own set of Data nodes and Action node configurations. This allows you to, for example, run the same sequence of FFT and plotting actions on different input TimeData nodes by simply switching the active Context. Shared "global" Data nodes can also be defined and accessed from multiple contexts if needed.

*Example*:
Imagine you have two datasets: "BaselineVibration.tdms" and "PostMaintenanceVibration.tdms".
- You could create a Context named "Baseline Analysis" where you load "BaselineVibration.tdms", perform an FFT, and plot the spectrum.
- Then, you could create another Context named "Post-Maintenance Analysis", load "PostMaintenanceVibration.tdms", and apply the same FFT and plotting actions, possibly with minor adjustments to parameters if needed.
This allows for a clear, side-by-side comparison while reusing the core analysis logic.

### Auxiliaries

Auxiliary tools enhance the usability and efficiency of the DAC framework:

*   **Quick Tasks (on Action nodes)**
    "Quick Tasks" are helper utilities associated with Action nodes, designed to simplify the process of configuring their parameters. Instead of manually typing in all parameters, a Quick Task can provide an interactive way to set them. For example, a "Load File" Action might have a Quick Task that opens a file dialog, allowing the user to browse and select a file, with the chosen path then automatically populating the Action's file path parameter. They streamline parameter input, especially for complex or frequently used configurations.

*   **Quick Actions (on Data nodes)**
    "Quick Actions" offer a way to perform rapid, exploratory analysis on Data nodes without formally adding new Action nodes to the project's workflow. When a Data node is selected, available Quick Actions (which are essentially pre-configured, lightweight versions of normal Actions) can be triggered. These actions operate on the selected data with default or minimal parameters. For instance, selecting a "TimeData" node might offer a Quick Action to quickly plot its content or calculate its RMS value. If the results are insightful and need to be part of the persistent analysis, the user can then create a full Action node. Quick Actions are ideal for ad-hoc data inspection and simple operations.

## Get started

This section guides you through installing and running the DAC application.

### Prerequisites

*   **Python**: DAC requires Python 3.10 or newer. You can download the latest version of Python from [python.org](https://www.python.org/downloads/).
*   **pip**: Ensure you have pip, the Python package installer, available. It usually comes with Python installations.

### Installation

It is highly recommended to install DAC in a virtual environment to avoid conflicts with other Python projects or your system's Python installation.

1.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv dac-env
    # On Windows
    # dac-env\Scripts\activate
    # On macOS and Linux
    # source dac-env/bin/activate
    ```

2.  **Install DAC from PyPI:**
    To install DAC including the GUI and its dependencies (`PyQt5`, `QScintilla`), run the following command:
    ```bash
    pip install miz-dac[gui]
    ```
    This will install all necessary core dependencies (`click`, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `nptdms`) as well as the GUI components.

    If you only intend to use DAC as a library without the GUI, you can install the core package:
    ```bash
    pip install miz-dac
    ```

### Running the Application

Once installed, you can run the DAC GUI application using the command provided by the package:

```bash
dac-bin
```

This will launch the main DAC application window.

Alternatively, DAC's components can be imported and used as a library in your own Python scripts or interactive sessions (e.g., Jupyter notebooks) if you have specific scripting needs. Refer to the module documentation for more details on programmatic usage (details to be added).

## Modules

DAC includes several specialized modules that extend its core functionality, providing tools and data types for common measurement data analysis tasks:

*   **`timedata`**: This is a fundamental module that provides the `TimeData` class for representing time-series data. It includes actions for loading data (e.g., from TDMS files via `data_loader.py`), basic processing (like truncating, filtering, resampling, and generating envelopes), signal construction from cosine components, and visualization of time-domain signals.

*   **`nvh` (Noise, Vibration, and Harshness)**: This module focuses on frequency-domain analysis. It defines data structures like `FreqDomainData` (for spectra) and `FreqIntermediateData` (for STFT-like results), along with `OrderList` and `OrderSliceData` for order analysis. Actions include FFT conversion, averaging, colormap visualization, order slice extraction, and spectral filtering. It also contains a `calc_lib.py` for common calculations like A-weighting.

*   **`drivetrain`**: Tailored for drivetrain analysis, this module provides classes for modeling components like `GearStage` (planetary and parallel) and `BallBearing`. It includes functionalities to calculate characteristic frequencies (e.g., gear mesh frequencies, bearing defect frequencies) and actions to create these component definitions and visualize their characteristic lines on time or frequency domain plots.

These modules build upon the core DAC framework, offering pre-built components to accelerate specific types of analysis.

## Extending DAC

DAC is designed to be extensible, allowing users to integrate their custom data types, processing algorithms, and visualization routines. This is primarily achieved by creating new Python modules and configuring their integration via YAML plugin files.

### 1. Creating a New Module

A new DAC module is typically a Python package (a directory containing an `__init__.py` file and other Python scripts). Inside your module, you'll usually define:

*   Custom Data types (often in a `data.py` file).
*   Custom Action types (often in an `actions.py` file).
*   Custom GUI Tasks for parameter input (optional, often in a `tasks.py` file).

**Module Structure Example:**
```
my_custom_module/
├── __init__.py
├── data.py       # Defines custom DataNode subclasses
├── actions.py    # Defines custom ActionNode subclasses
└── tasks.py      # Optional: Defines custom TaskBase subclasses for GUI interaction
```

Your `__init__.py` should make your custom classes easily importable, e.g.:
```python
# my_custom_module/__init__.py
from .data import MyCustomData, AnotherDataStructure
from .actions import MyCustomAction, MyVisualizationAction
# from .tasks import MyCustomGUITask # if you have tasks
```

### 2. Defining Custom Data Types (`data.py`)

Custom data types should inherit from `dac.core.DataNode` or one of its more specialized children (like `dac.core.data.DataBase` if it's a primary data container, or `dac.core.ContextKeyNode` if it's to be used as a context type).

**Key aspects for a custom `DataNode`:**

*   **`__init__(self, name: str = None, uuid: str = None, ...)`**: Initialize your data attributes.
*   **`get_construct_config(self) -> dict`**: This method is crucial. It must return a dictionary representing the state of your DataNode that needs to be saved and can be edited in the GUI. Keys should typically match your `__init__` parameters. Values should be basic Python types (strings, numbers, lists, dicts) or representations that can be easily serialized to YAML/JSON.
*   **`apply_construct_config(self, construct_config: dict)`**: This method takes a dictionary (usually from loading a saved project or from the GUI editor) and applies it to the DataNode's attributes.
*   **(Optional) `QUICK_ACTIONS: list[tuple[type[ActionBase], str, dict]]`**: A class attribute. If you want to define "Quick Actions" (right-click context menu actions in the GUI's data list) for this specific data type, you can list them here. Each tuple specifies:
    1.  The `ActionNode` class to execute.
    2.  The name of the parameter in the Action's `__call__` method that should receive the selected DataNode(s).
    3.  A dictionary of other default parameters for the Action.

**Example (`my_custom_module/data.py`):**
```python
from dac.core.data import DataBase
# from ..nvh.actions import ViewFreqDomainAction # Example Quick Action

class MySpectrum(DataBase):
    def __init__(self, name: str = None, uuid: str = None, frequencies: list = None, amplitudes: list = None, unit: str = "EU"):
        super().__init__(name, uuid)
        self.frequencies = frequencies if frequencies is not None else []
        self.amplitudes = amplitudes if amplitudes is not None else []
        self.unit = unit

    def get_construct_config(self) -> dict:
        return {
            "name": self.name,
            # For large data like frequencies/amplitudes, you might store them externally
            # or return a summary/placeholder if they are not directly editable.
            # Here, we assume they are not directly edited in the node editor for simplicity.
            "unit": self.unit,
            "num_points": len(self.frequencies)
        }

    def apply_construct_config(self, construct_config: dict):
        self.name = construct_config.get("name", self.name)
        self.unit = construct_config.get("unit", self.unit)
        # Frequencies/amplitudes would typically be set by an Action, not here.

    # Example Quick Action (assuming ViewFreqDomainAction can plot this)
    # QUICK_ACTIONS = [
    #     (ViewFreqDomainAction, "channels", {"with_phase": False})
    # ]
```

### 3. Defining Custom Actions (`actions.py`)

Custom actions define the processing steps or visualizations. They should inherit from `dac.core.ActionNode` or its more specific children:
*   `dac.core.actions.ProcessActionBase` (PAB): For actions that perform data processing, potentially long-running, and may benefit from running in a separate thread (handled by the GUI).
*   `dac.core.actions.VisualizeActionBase` (VAB): For actions that generate Matplotlib plots in the GUI.
*   `dac.core.actions.SequenceActionBase` (SAB): For creating a fixed sequence of other actions.

**Key aspects for a custom `ActionNode`:**

*   **`CAPTION: str`**: A class attribute defining the user-friendly name of the action displayed in the GUI.
*   **`__call__(self, ...)`**: The core method that performs the action.
    *   **Input Parameters**: Define them as arguments to `__call__`. Their type hints are crucial:
        *   If an argument is type-hinted with a `DataNode` subclass (e.g., `my_data: MySpectrum`), the GUI will attempt to link a `MySpectrum` node from the current context (or global nodes if applicable) to this parameter. The value passed will be the actual `MySpectrum` instance.
        *   Other type hints (e.g., `int`, `float`, `str`, `list`, `Enum`) will be presented as configurable fields in the GUI editor. Default values can be provided.
    *   **Output**: The return value of `__call__` (if any) will typically be a new `DataNode` instance (or a list of them). The GUI will add this to the current context. The `out_name` property of the action can be used to suggest a name for the output node.
*   **(Optional) `_SIGNATURE: inspect.Signature`**: Usually automatically determined from `__call__`. For complex cases like `SequenceActionBase`, it might be defined manually.
*   **(Optional) `QUICK_TASKS: list[TaskBase]`**: A class attribute listing `TaskBase` instances that can help fill parameters for this Action in the GUI.
*   **(Optional) `DEFAULT_TASK: TaskBase`**: A default `TaskBase` to be invoked.

**Example (`my_custom_module/actions.py`):**
```python
from dac.core.actions import ProcessActionBase
from .data import MySpectrum # Your custom data type
import numpy as np

class CalculateMeanAmplitude(ProcessActionBase):
    CAPTION = "Calculate Mean Amplitude"

    def __call__(self, spectrum: MySpectrum, freq_min: float = 0, freq_max: float = 1000) -> dict: # Output can be a simple dict for summary
        """
        Calculates the mean amplitude of MySpectrum within a frequency range.
        """
        indices = np.where((np.array(spectrum.frequencies) >= freq_min) & (np.array(spectrum.frequencies) <= freq_max))
        if not indices[0].size:
            mean_amp = 0
        else:
            mean_amp = np.mean(np.array(spectrum.amplitudes)[indices])
        
        # This action outputs a dictionary, which isn't directly a DataNode.
        # A more complete DAC integration might involve creating a new DataNode type for results,
        # or using a generic 'ResultData' node. For simplicity, we output a dict.
        # The GUI might not directly show this dict unless a specific view for it is made.
        # Often, outputs are new DataNode instances.
        self.message(f"Calculated mean amplitude: {mean_amp} {spectrum.unit}")
        return {"mean_amplitude": mean_amp, "unit": spectrum.unit, "range": [freq_min, freq_max]}

class MySimpleVisAction(VisualizeActionBase): # Example VAB
    CAPTION = "Plot MySpectrum"

    def __call__(self, spec_data: MySpectrum):
        ax = self.figure.gca()
        ax.plot(spec_data.frequencies, spec_data.amplitudes)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude ({spec_data.unit})")
        ax.set_title(spec_data.name)
```

### 4. Registering and Configuring with `plugins.yaml`

After defining your custom data and action classes, you need to inform the DAC application about them using a YAML plugin file (e.g., `my_plugin.yaml`). The application loads these YAML files to populate the lists of available data types and actions in the GUI.

**Key YAML sections:**

*   **`alias`**: (Optional) Define short aliases for long module paths.
    ```yaml
    alias:
      my_mod: "my_project.my_custom_module"
    ```
*   **`data`**: Register global data types that can be used as `ContextKeyNode`s (i.e., types that can define a context).
    ```yaml
    data:
      _: # Global context data types
        - "[My Custom Data Types]" # Section separator string
        - "my_mod.data.MyProjectContextData" # Example: a DataNode for project-level context
    ```
*   **`actions`**: This is the most important section for making your actions available.
    *   Use `_` for actions available in the global context.
    *   Use the full Python path to your `ContextKeyNode` subclass (e.g., `dac.core.data.SimpleDefinition` or your custom context data type) to define actions specific to that context.
    ```yaml
    actions:
      _: # Actions available in the Global Context (GCK)
        - "[My Global Actions]" # Section separator
        - "my_mod.actions.MyGlobalUtilityAction"
      dac.core.data.SimpleDefinition: # Actions for the default SimpleDefinition context
        - "[My Analysis Actions]"
        - "my_mod.actions.CalculateMeanAmplitude"
        - "my_mod.actions.MySimpleVisAction"
      # "my_mod.data.MyProjectContextData": # Actions specific to MyProjectContextData context
      #   - "my_mod.actions.AnotherSpecificAction"
    ```
    You can create menu hierarchies using section separators: `[Section Name>]` to start a submenu and `[<Section Name]` to end it.
*   **`quick_tasks`**: (Optional) Link `TaskBase` implementations (from your `tasks.py`) to specific `ActionNode` classes for GUI-assisted parameter input.
    ```yaml
    quick_tasks:
      "my_mod.actions.MyDataLoadingAction": # The Action class
        - ["my_mod.tasks.MyFileBrowserTask", "Browse for file...", "fpath"] # Task class, button name, target param name
    ```
*   **`quick_actions`**: (Optional) Register quick actions for your custom `DataNode` types.
    ```yaml
    quick_actions:
      "my_mod.data.MySpectrum": # The DataNode class
        # (ActionClass, data_param_name_in_action, {other_default_params})
        - ["my_mod.actions.MySimpleVisAction", "spec_data", {}]
        - ["my_mod.actions.CalculateMeanAmplitude", "spectrum", {"freq_max": 500}]
    ```

The DAC GUI loads plugins from the `dac/plugins` directory by default. You can place your YAML file there or load it programmatically.

### 5. Workflow Summary for Extending

1.  **Plan your module**: Decide on the new data types and actions you need.
2.  **Implement Data Nodes**: Create subclasses of `DataNode` in your module's `data.py`. Implement `__init__`, `get_construct_config`, and `apply_construct_config`.
3.  **Implement Action Nodes**: Create subclasses of `ActionNode` (or PAB, VAB, SAB) in your module's `actions.py`. Define `CAPTION` and the `__call__` method with appropriate type hints.
4.  **Create Plugin YAML**: Create or update a `.yaml` file to register your new data types (if they are context keys) and actions. Define which contexts your actions should appear in. Optionally add quick tasks and quick actions.
5.  **Test**: Launch the DAC application. Your new actions should appear in the "Add Action" menus of the specified contexts, and your data types should be usable.

By following this structure, you can integrate custom functionality seamlessly into the DAC framework. Remember to consult the existing `dac.core` and standard modules (`timedata`, `nvh`, `drivetrain`) for more examples.

## Appendix

### Design Philosophy: OOP and Functional Aspects

DAC employs an Object-Oriented Programming (OOP) approach for its core components: `DataNode` and `ActionNode` subclasses. This design promotes:
*   **Encapsulation**: Data and its associated metadata are encapsulated within Data nodes. Operations and their parameters are encapsulated within Action nodes.
*   **Modularity**: New data types and operations can be added as new classes, making the system extensible.
*   **Reusability**: Defined Data and Action nodes can be reused across different analyses and contexts.

While the framework structure is object-oriented (nodes connected in a graph), the actual data processing logic within an `ActionNode`'s `__call__` method is often implemented using functional programming principles or by calling library functions (e.g., NumPy, SciPy operations). This combination allows for a structured and extensible framework while leveraging the power of functional libraries for computations.

### Contributing Guidelines

We welcome contributions to improve and expand DAC! If you'd like to contribute, please follow these general guidelines:

1.  **Check Existing Issues**: Before starting new work, please check the [GitHub Issues](https://github.com/MIZiper/dac/issues) to see if your idea or bug is already being discussed.
2.  **Fork and Branch**: Fork the repository and create a new branch for your feature or bugfix (e.g., `feature/my-new-feature` or `bugfix/fix-that-bug`).
3.  **Develop**:
    *   Write clear and concise code.
    *   Follow the general coding style and conventions found in the project. You can refer to `doc/code_convention.py` for examples of naming and structure.
    *   Ensure your changes are well-documented with docstrings (NumPy format preferred for new contributions).
4.  **Test**:
    *   Add unit tests for any new functionality or bug fixes.
    *   Ensure all tests pass before submitting your changes. (Note: Specific instructions on running tests would ideally be here if a test suite is set up).
5.  **Pull Request**: Submit a pull request to the `main` branch (or the relevant development branch).
    *   Provide a clear title and a detailed description of your changes.
    *   Link to any relevant issues.

### License

DAC is licensed under the Apache License 2.0. You can find the full license text in the `LICENSE` file in the repository (though one isn't explicitly listed by `ls`, it's standard practice, or implied by `pyproject.toml`). The license information is also specified in the `pyproject.toml` file.

### Reporting Issues and Support

*   **Bug Reports & Feature Requests**: Please submit them through the [GitHub Issues](https://github.com/MIZiper/dac/issues) page for the repository.
*   **General Contact**: For other inquiries, you can reach out to the author, MIZiper, via email at `miziper@163.com` (as listed in `pyproject.toml`).
