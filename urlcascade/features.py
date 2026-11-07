"""The 18 numeric inputs in the order expected by the pinned GRU."""
import re
from urllib.parse import urlparse

FEATURE_NAMES = (
    'use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@', 'count_dir',
    'count_embed_domian', 'count-https', 'count-http', 'count%', 'count?',
    'count-', 'count=', 'url_length', 'hostname_length', 'count-digits',
    'count-letters', 'fd_length',
)

# The reference also recognizes a hexadecimal IPv4 followed immediately by
# a full IPv6 after the slash. An ordinary IPv6 on its own does not match.
_octet = r'(?:[01]?\d\d?|2[0-4]\d|25[0-5])'
_hex_byte = r'0x[0-9a-fA-F]{1,2}'
_hex_word = r'[0-9a-fA-F]{1,4}'
_address = re.compile(
    r'\.'.join([_octet] * 4) + r'/|' +
    r'\.'.join([_hex_byte] * 4) + '/' + ':'.join([_hex_word] * 8)
)


def extract_features(raw_url):
    parsed = urlparse(raw_url)
    parts = parsed.path.split('/', 2)
    # The original uses the hostname as a regex, including its metacharacters.
    return [
        int(bool(_address.search(raw_url))),
        int(bool(re.search(str(parsed.hostname), raw_url))),
        raw_url.count('.'), raw_url.count('www'), raw_url.count('@'),
        parsed.path.count('/'), parsed.path.count('//'),
        raw_url.count('https'), raw_url.count('http'), raw_url.count('%'),
        raw_url.count('?'), raw_url.count('-'), raw_url.count('='),
        len(raw_url), len(parsed.netloc), sum(map(str.isnumeric, raw_url)),
        sum(map(str.isalpha, raw_url)), len(parts[1]) if len(parts) > 1 else 0,
    ]
