import traceback, sys, inspect
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, pyqtSlot, QMutex

# https://gist.github.com/ksvbka/1f26ada0c6201c2cf19f59b100d224a9

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    - started: No data
    - finished: No data
    - error: `exctype, value, traceback`
    - result: `object` data returned from processing, anything
    - progress: `int, int` indicating progress metadata
    - message: `str` the message from worker
    '''

    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(type, object, object)
    result = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    message = pyqtSignal(str)

class ThreadWorker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    '''

    def __init__(self, fn, caption: str, *args, mutex: QMutex=None, **kwargs):
        super().__init__()
        self.caption = caption
        self.mutex = mutex

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        
        # Add the callback to our kwargs
        fn_params = inspect.signature(self.fn).parameters
        if 'progress_emitter' in fn_params:
            self.kwargs['progress_emitter'] = self.signals.progress.emit
        if 'logger' in fn_params:
            self.kwargs['logger'] = self.signals.message.emit

        try:
            if self.mutex is not None:
                self.mutex.lock()
            self.signals.started.emit()
            
            result = self.fn(*self.args, **self.kwargs)
        except:
            # traceback.print_exc()
            # exctype, value, tb = sys.exc_info()
            self.signals.error.emit(*sys.exc_info())
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            if self.mutex is not None:
                self.mutex.unlock()
            self.signals.finished.emit()  # Done