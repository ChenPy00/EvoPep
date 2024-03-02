import numpy as np

RESIDUES = tuple('ACDEFGHIKLMNPQRSTVWY')


class ResidueIdentityEncoder(object):
    """
    Residue identity (one-hot) encoder.
    Attributes:
        encoding_size: (int) The number of encoding dimensions for a single residue.
    """

    def __init__(self, alphabet, place_holder='_'):
        """
        Constructor.
        Args:
            alphabet: (seq<char>) The alphabet of valid tokens for the sequence;
        e.g., the 20 x 1-letter residue codes for standard peptides.
        """
        self._alphabet = [l.upper() for l in alphabet]
        self._letter_to_id = dict((letter, id) for (id, letter) in enumerate(self._alphabet))
        self._encoding_size = len(self._alphabet)
        self._place_holder = place_holder

    def __call__(self, residue):
        """
        Encodes a single residue as a one hot identity vector.
        Args:
            residue: (str) A single-character string representing one residue;
        e.g.:
            'A' for Alanine.
        Returns:
            A numpy.ndarray(shape=(A,), dtype=float) with a single non-zero (1) value;
            the identity index for each residue in the alphabet is given by the
            residue's index in the alphabet ordered sequence; i.e., for the alphabet
            'ACDE', a 'C' would be encoded as [0, 1, 0, 0].
        """
        onehot = np.zeros(self._encoding_size, dtype=float)
        if residue != self._place_holder:
            onehot[self._letter_to_id[residue]] = 1
        return onehot


class DirectSequenceEncoder(object):
    """
    Direct sequence encoder for generating variable-length representations.
    Attributes:
        encoding_size: (int) The encoding length for a single residue.
    e.g.:
        SequenceEncoder = DirectSequenceEncoder(ResidueIdentityEncoder(RESIDUES))
    """

    def __init__(self, residue_encoder):
        """
        Constructor.
        Args:
            residue_encoder: (object) A single residue encoder.
        """
        self._residue_encoder = residue_encoder
        self._encoding_size = self._residue_encoder._encoding_size

    def __call__(self, seq):
        """
        Encodes a sequence as a variable-length multi-dimensional array.
        Args:
            seq: (str) A residue sequence to encode; e.g., "ATEST".
        Returns:
            (numpy.ndarray(shape=(len(seq), encoding_size), dtype=float)) The encoded residue sequence.
        """
        return np.array([self._residue_encoder(r) for r in seq.upper()])
