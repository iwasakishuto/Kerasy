# coding: utf-8
import hashlib
from struct import unpack, pack, calcsize

def optimize_bit_format(bits_size):
    """
    By default, C types are represented in the machine’s native format and byte
    order. Alternatively, the first character of the format string can be used
    to indicate the byte order, size and alignment of the packed data.
    ~~~
    @params bits_size : Decimal integer.
    @return fmt_code  : Format codes means the conversion between C and Python values.
                        Ref: https://docs.python.org/3/library/struct.html#format-characters
    @return std_size  : Refers to the size of the packed value in bytes when using standard size.
    """
    if bits_size >= (1 << 31):
        fmt_code, std_size = 'Q', 8 # unsigned long long
    elif bits_size >= (1 << 15):
        fmt_code, std_size = 'I', 4 # long
    else:
        fmt_code, std_size = 'H', 2 # unsigned short
    return (fmt_code, std_size)

def optimize_hash_func(bits_size):
    """
    @params bits_size : Decimal integer.
    @return hash_func : Optimized hash function.
    """
    if bits_size > 384:
        hash_func = hashlib.sha512
    elif bits_size > 256:
        hash_func = hashlib.sha384
    elif bits_size > 160:
        hash_func = hashlib.sha256
    elif bits_size > 128:
        hash_func = hashlib.sha1
    else:
        hash_func = hashlib.md5
    return hash_func

def make_hashfuncs_partition(bits_per_slice, num_slices):
    """
    @params bits_per_slice : The bit size for one hash function.
    @params num_slices     : The number of hash function.
    ~~~
    Reference:
    * Github: https://github.com/jaybaird/python-bloomfilter/blob/2bbe01ad49965bf759e31781e6820408068862ac/pybloom/pybloom.py#L54
    * Paper : F. Chang, W. chang Feng, K. Li, "Approximate caches for packet classification".
    """
    fmt_code, chunk_size = optimize_bit_format(bits_per_slice)
    total_hash_bits = num_slices * (8*chunk_size)
    hash_func = optimize_hash_func(total_hash_bits)
    fmt = fmt_code * (hash_func().digest_size // chunk_size)
    num_salts, extra = divmod(num_slices, len(fmt))
    if extra:
        num_salts += 1
    salts = tuple(hash_func(hash_func(pack('I', i)).digest()) for i in range(num_salts))

    def _make_hashfuncs(key):
        key = str(key).encode('utf-8')
        i = 0
        for salt in salts:
            h = salt.copy()
            h.update(key)
            for uint in unpack(fmt, h.digest()):
                yield uint % bits_per_slice
                i += 1
                if i >= num_slices:
                    return
    return _make_hashfuncs


def make_hashfuncs(bits_size, num_hashes, method="partition"):
    """
    @params bits_size  : (int) total bits size.
    @params num_hashes : (int) The number of hash function.
    @params method     : (str)
        - partition : consists of partitioning the M bits among the k hash functions.
    """
    if method=="partition":
        if bits_size%num_hashes != 0:
            raise ValueError(
                f"If you use `partition` method, bits_size%num_hashes must be 0, \
                but got {bits_size%num_hashes}."
            )
        bits_per_slice = bits_size//num_hashes
        _make_hashfuncs = make_hashfuncs_partition(bits_per_slice, num_hashes)
        
    return _make_hashfuncs
