# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# # Import the low-level C/C++ module
# if __package__ or "." in __name__:
#     from . import _impl
# else:
import _impl

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class TokenMetadata(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    text = property(_impl.TokenMetadata_text_get)
    timestep = property(_impl.TokenMetadata_timestep_get)
    start_time = property(_impl.TokenMetadata_start_time_get)

    def __repr__(self):
      return 'TokenMetadata(text=\'{}\', timestep={}, start_time={})'.format(self.text, self.timestep, self.start_time)


# Register TokenMetadata in _impl:
_impl.TokenMetadata_swigregister(TokenMetadata)

class CandidateTranscript(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    tokens = property(_impl.CandidateTranscript_tokens_get)
    confidence = property(_impl.CandidateTranscript_confidence_get)

    def __repr__(self):
      tokens_repr = ',\n'.join(repr(i) for i in self.tokens)
      tokens_repr = '\n'.join('  ' + l for l in tokens_repr.split('\n'))
      return 'CandidateTranscript(confidence={}, tokens=[\n{}\n])'.format(self.confidence, tokens_repr)


# Register CandidateTranscript in _impl:
_impl.CandidateTranscript_swigregister(CandidateTranscript)

class Metadata(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    transcripts = property(_impl.Metadata_transcripts_get)

    def __repr__(self):
      transcripts_repr = ',\n'.join(repr(i) for i in self.transcripts)
      transcripts_repr = '\n'.join('  ' + l for l in transcripts_repr.split('\n'))
      return 'Metadata(transcripts=[\n{}\n])'.format(transcripts_repr)

    __swig_destroy__ = _impl.delete_Metadata

# Register Metadata in _impl:
_impl.Metadata_swigregister(Metadata)

ERR_OK = _impl.ERR_OK
ERR_NO_MODEL = _impl.ERR_NO_MODEL
ERR_INVALID_ALPHABET = _impl.ERR_INVALID_ALPHABET
ERR_INVALID_SHAPE = _impl.ERR_INVALID_SHAPE
ERR_INVALID_SCORER = _impl.ERR_INVALID_SCORER
ERR_MODEL_INCOMPATIBLE = _impl.ERR_MODEL_INCOMPATIBLE
ERR_SCORER_NOT_ENABLED = _impl.ERR_SCORER_NOT_ENABLED
ERR_SCORER_UNREADABLE = _impl.ERR_SCORER_UNREADABLE
ERR_SCORER_INVALID_LM = _impl.ERR_SCORER_INVALID_LM
ERR_SCORER_NO_TRIE = _impl.ERR_SCORER_NO_TRIE
ERR_SCORER_INVALID_TRIE = _impl.ERR_SCORER_INVALID_TRIE
ERR_SCORER_VERSION_MISMATCH = _impl.ERR_SCORER_VERSION_MISMATCH
ERR_FAIL_INIT_MMAP = _impl.ERR_FAIL_INIT_MMAP
ERR_FAIL_INIT_SESS = _impl.ERR_FAIL_INIT_SESS
ERR_FAIL_INTERPRETER = _impl.ERR_FAIL_INTERPRETER
ERR_FAIL_RUN_SESS = _impl.ERR_FAIL_RUN_SESS
ERR_FAIL_CREATE_STREAM = _impl.ERR_FAIL_CREATE_STREAM
ERR_FAIL_READ_PROTOBUF = _impl.ERR_FAIL_READ_PROTOBUF
ERR_FAIL_CREATE_SESS = _impl.ERR_FAIL_CREATE_SESS
ERR_FAIL_CREATE_MODEL = _impl.ERR_FAIL_CREATE_MODEL
ERR_FAIL_INSERT_HOTWORD = _impl.ERR_FAIL_INSERT_HOTWORD
ERR_FAIL_CLEAR_HOTWORD = _impl.ERR_FAIL_CLEAR_HOTWORD
ERR_FAIL_ERASE_HOTWORD = _impl.ERR_FAIL_ERASE_HOTWORD

def CreateModel(aModelPath):
    return _impl.CreateModel(aModelPath)

def GetModelBeamWidth(aCtx):
    return _impl.GetModelBeamWidth(aCtx)

def SetModelBeamWidth(aCtx, aBeamWidth):
    return _impl.SetModelBeamWidth(aCtx, aBeamWidth)

def GetModelSampleRate(aCtx):
    return _impl.GetModelSampleRate(aCtx)

def FreeModel(ctx):
    return _impl.FreeModel(ctx)

def EnableExternalScorer(aCtx, aScorerPath):
    return _impl.EnableExternalScorer(aCtx, aScorerPath)

def AddHotWord(aCtx, word, boost):
    return _impl.AddHotWord(aCtx, word, boost)

def EraseHotWord(aCtx, word):
    return _impl.EraseHotWord(aCtx, word)

def ClearHotWords(aCtx):
    return _impl.ClearHotWords(aCtx)

def DisableExternalScorer(aCtx):
    return _impl.DisableExternalScorer(aCtx)

def SetScorerAlphaBeta(aCtx, aAlpha, aBeta):
    return _impl.SetScorerAlphaBeta(aCtx, aAlpha, aBeta)

def SpeechToText(aCtx, aBuffer):
    return _impl.SpeechToText(aCtx, aBuffer)

def SpeechToTextWithMetadata(aCtx, aBuffer, aNumResults):
    return _impl.SpeechToTextWithMetadata(aCtx, aBuffer, aNumResults)

def CreateStream(aCtx):
    return _impl.CreateStream(aCtx)

def FeedAudioContent(aSctx, aBuffer):
    return _impl.FeedAudioContent(aSctx, aBuffer)

def IntermediateDecode(aSctx):
    return _impl.IntermediateDecode(aSctx)

def IntermediateDecodeWithMetadata(aSctx, aNumResults):
    return _impl.IntermediateDecodeWithMetadata(aSctx, aNumResults)

def FinishStream(aSctx):
    return _impl.FinishStream(aSctx)

def FinishStreamWithMetadata(aSctx, aNumResults):
    return _impl.FinishStreamWithMetadata(aSctx, aNumResults)

def FreeStream(aSctx):
    return _impl.FreeStream(aSctx)

def FreeMetadata(m):
    return _impl.FreeMetadata(m)

def FreeString(str):
    return _impl.FreeString(str)

def Version():
    return _impl.Version()

def ErrorCodeToErrorMessage(aErrorCode):
    return _impl.ErrorCodeToErrorMessage(aErrorCode)


