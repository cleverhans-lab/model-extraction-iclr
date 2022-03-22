#!/usr/bin/env python3
# !/usr/bin/env python2.3
"""Implement Hashcash version 1 protocol in Python
+-------------------------------------------------------+
| Written by David Mertz; released to the Public Domain |
+-------------------------------------------------------+
Edited for use with Python 3.

Double spend database not implemented in this module, but stub
for callbacks is provided in the 'check()' function

The function 'check()' will validate hashcash v1 and v0 tokens, as well as
'generalized hashcash' tokens generically.  Future protocol version are
treated as generalized tokens (should a future version be published w/o
this module being correspondingly updated).

A 'generalized hashcash' is implemented in the '_mint()' function, with the
public function 'mint()' providing a wrapper for actual hashcash protocol.
The generalized form simply finds a suffix that creates zero bits in the
hash of the string concatenating 'challenge' and 'suffix' without specifying
any particular fields or delimiters in 'challenge'.  E.g., you might get:

>>> mint('foo', bits=16)
'1:16:210707:foo::FjIKxj==:60b5'

>>> _mint('foo', bits=16)
'9591'
>>> from hashlib import sha1 as sha
>>> m = sha()
>>> m.update(b'foo9591')
>>> m.hexdigest()
'0000de4c9b27cec9b20e2094785c1c58eaf23948'
>>> m = sha()
>>> m.update(b'1:16:040922:foo::+ArSrtKd:164b3')
>>> m.hexdigest()
'0000a9fe0c6db2efcbcab15157735e77c0877f34'

Notice that '_mint()' behaves deterministically, finding the same suffix
every time it is passed the same arguments.  'mint()' incorporates a random
salt in stamps (as per the hashcash v.1 protocol).
"""
from string import ascii_letters
import binascii
import sys
from hashlib import sha1 as sha
from math import ceil, floor
from random import choice
from time import strftime, localtime, time

ERR = sys.stderr  # Destination for error messages
DAYS = 60 * 60 * 24  # Seconds in a day
tries = [0]  # Count hashes performed for benchmark


def generate_challenge(resource, bits=20, now=None, ext='', saltchars=8,
                       stamp_seconds=False):
    """
    Server challenge function.

    Args:
        resource: a unique name of the resource, can be an email address or a
            domain name.
        bits: number of required leading zeros. This is proportional to the
            amount of work that a client has to expand on average to mint a
            token.
        now: a time in seconds since the Epoch.
        ext: your own extensions to the minted stamp
        saltchars: specifies the length of the salt used; this version defaults
            8 chars
        stamp_seconds: lets you add the option time elements to the datestamp.
            If you want more than just day, you get all the way down to seconds.

    Returns:
        The challenge string.

    """
    ver = "1"
    now = now or time()
    if stamp_seconds:
        ts = strftime("%y%m%d%H%M%S", localtime(now))
    else:
        ts = strftime("%y%m%d", localtime(now))
    challenge = "%s:" * 6 % (ver, bits, ts, resource, ext, _salt(saltchars))
    return challenge


def mint(resource, bits=20, now=None, ext='', saltchars=8, stamp_seconds=False,
         xtype='bin'):
    """Mint a new hashcash stamp for 'resource' with 'bits' of collision

    20 bits of collision is the default.

    'ext' lets you add your own extensions to a minted stamp.  Specify an
    extension as a string of form 'name1=2,3;name2;name3=var1=2,2,val'
    FWIW, urllib.urlencode(dct).replace('&',';') comes close to the
    hashcash extension format.

    'saltchars' specifies the length of the salt used; this version defaults
    8 chars, rather than the C version's 16 chars.  This still provides about
    17 million salts per resource, per timestamp, before birthday paradox
    collisions occur.  Really paranoid users can use a larger salt though.

    'stamp_seconds' lets you add the option time elements to the datestamp.
    If you want more than just day, you get all the way down to seconds,
    even though the spec also allows hours/minutes without seconds.

    'xtype' is the representation type (hex or bin) used to check for the
    leading zeros.
    """
    challenge = generate_challenge(
        resource=resource, bits=bits, now=now, ext=ext, saltchars=saltchars,
        stamp_seconds=stamp_seconds)
    return mint_iteractive(challenge=challenge, bits=bits, xtype=xtype)


def mint_iteractive(challenge, bits=20, xtype='bin'):
    return challenge + _mint(challenge=challenge, bits=bits, xtype=xtype)


def _salt(string_length):
    """
    Return a random string of length 'string_length'.
    """
    alphabet = ascii_letters + "+/="
    return ''.join([choice(alphabet) for _ in [None] * string_length])


def _get_digest(input):
    """

    Args:
        input: input string

    Returns: secure hash of the byte representation of the input

    """
    m = sha()
    input = input.encode('utf-8')
    m.update(input)
    return m.hexdigest()


def _mint(challenge, bits, xtype='bin'):
    """Answer a 'generalized hashcash' challenge'

    Mint token based on the challenge.

    Hashcash requires stamps of form 'ver:bits:date:res:ext:rand:counter'
    This internal function accepts a generalized prefix 'challenge',
    and returns only a suffix that produces the requested SHA leading zeros.

    NOTE: For hexadecimal representation type, the number of requested bits is
    rounded up to the nearest multiple of 4
    """
    counter = 0

    # Separate cases for hex and bin to limit the comparisons in the main while
    # loop.
    if xtype == 'hex':
        hex_digits = int(ceil(bits / 4.))
        zeros = '0' * hex_digits
        while 1:
            # 2: omit standard 0x prefix in the hex representation.
            suffix = hex(counter)[2:]
            digest = _get_digest(input=challenge + suffix)
            if digest[:hex_digits] == zeros:
                tries[0] = counter
                print('hex digest: ', digest)
                print('bin digest: ', _to_binary(digest))
                return suffix
            counter += 1
            # print('counter: ', counter)
    else:
        zeros = '0' * bits
        while 1:
            # 2: omit standard 0x prefix in the hex representation.
            suffix = hex(counter)[2:]
            digest = _get_digest(input=challenge + suffix)
            bin_digest = _to_binary(digest)
            if bin_digest[:bits] == zeros:
                tries[0] = counter
                return suffix
            counter += 1


def check(stamp, resource=None, bits=None,
          check_expiration=None, ds_callback=None, stamp_seconds=False,
          xtype='bin'):
    """Check whether a stamp is valid

    Optionally, the stamp may be checked for a specific resource, and/or
    it may require a minimum bit value, and/or it may be checked for
    expiration, and/or it may be checked for double spending.

    If 'check_expiration' is specified, it should contain the number of
    seconds old a date field may be.  Indicating days might be easier in
    many cases, e.g.

    >>> check(stamp, check_expiration=28*DAYS)

    NOTE: Every valid (version 1) stamp must meet its claimed bit value
    NOTE: Check floor of 4-bit multiples (overly permissive in acceptance)
    """
    if stamp.startswith('0:'):  # Version 0
        try:
            # 2: omit the version number in the stamp
            date, res, suffix = stamp[2:].split(':')
        except ValueError:
            ERR.write("Malformed version 0 hashcash stamp!\n")
            return False
        if resource is not None and resource != res:
            return False
        elif check_expiration is not None:
            if stamp_seconds:
                good_until = strftime("%y%m%d%H%M%S",
                                      localtime(time() - check_expiration))
            else:
                good_until = strftime("%y%m%d",
                                      localtime(time() - check_expiration))
            if date < good_until:
                return False
        elif callable(ds_callback) and ds_callback(stamp):
            return False
        elif type(bits) is not int:
            return True

        hex_digits = int(floor(bits / 4))
        return sha(stamp).hexdigest().startswith('0' * hex_digits)

    elif stamp.startswith('1:'):  # Version 1
        try:
            # 2: omit the version number in the stamp
            claim, date, res, ext, rand, counter = stamp[2:].split(':')
        except ValueError:
            ERR.write("Malformed version 1 hashcash stamp!\n")
            return False
        if resource is not None and resource != res:
            return False
        elif type(bits) is int and bits > int(claim):
            return False
        elif check_expiration is not None:
            if stamp_seconds:
                good_until = strftime("%y%m%d%H%M%S",
                                      localtime(time() - check_expiration))
            else:
                good_until = strftime("%y%m%d",
                                      localtime(time() - check_expiration))

            if date < good_until:
                return False
        elif callable(ds_callback) and ds_callback(stamp):
            return False

        digest = _get_digest(input=stamp)
        if xtype == 'hex':
            hex_digits = int(floor(int(claim) / 4))
            return digest.startswith('0' * hex_digits)
        else:
            digest = _to_binary(digest)
            return digest.startswith('0' * int(claim))

    else:  # Unknown ver or generalized hashcash
        ERR.write("Unknown hashcash version: Minimal authentication!\n")
        if type(bits) is not int:
            return True
        elif resource is not None and stamp.find(resource) < 0:
            return False
        else:
            hex_digits = int(floor(bits / 4))
            return sha(stamp).hexdigest().startswith('0' * hex_digits)


def _to_binary(hexdata):
    scale = 16  # equals to hexadecimal
    num_of_bits = 160  # 160 bits for sha1
    # 2: discard 0b prefix.
    # zfill restores the leading zeros that we need.
    return bin(int(hexdata, scale))[2:].zfill(num_of_bits)


def is_doublespent(stamp):
    """Placeholder for double spending callback function.
    This is needed for the non-interactive mode. In the interactive mode, the
    stamps are randomly generated by the server.

    The check() function may accept a 'ds_callback' argument, e.g.
      check(stamp, "mertz@gnosis.cx", bits=20, ds_callback=is_doublespent)

    This placeholder simply reports stamps as not being double spent.
    """
    return False


if __name__ == '__main__':
    import argparse

    out, err = sys.stdout.write, sys.stderr.write
    parser = argparse.ArgumentParser(
        description="version 0.1",
        usage="python hashcash.py -c|-m [-b bits] [string|STDIN]")
    parser.add_argument('-b', '--bits', type=int, dest='bits', default=20,
                        help="Specify required collision bits")
    parser.add_argument('-r', '--resource', type=str, dest='resource',
                        default="model-extraction-warning",
                        help="Specify the name of the resource")
    parser.add_argument('-m', '--mint', help="Mint a new stamp",
                        action='store_true', dest='mint')
    parser.add_argument('-c', '--check', help="Check a stamp for validity",
                        action='store_true', dest='check')
    parser.add_argument('-t', '--timer', help="Time the operation performed",
                        action='store_true', dest='timer')
    parser.add_argument('-n', '--raw', help="Suppress trailing newline",
                        action='store_true', dest='raw')
    parser.add_argument('-s', '--stamp', help="The stamp for the check.",
                        dest='stamp', default=None)
    parser.add_argument('-x', '--xtype', type=str,
                        help="The type of the representation "
                             "either in hex or in binary.",
                        dest='xtype', default='bin')
    args = parser.parse_args()
    start = time()
    if args.mint:
        first_arg = args.resource
        out(str(mint(resource=args.resource, bits=args.bits)))
    elif args.check:
        if args.stamp is None:
            raise Exception(
                'The -s (--stamp) has to be provided for the check.')
        out(str(check(stamp=args.stamp, bits=args.bits, xtype=args.xtype)))
    else:
        out("Try: %s --help\n" % sys.argv[0])
        sys.exit()
    if not args.raw:
        sys.stdout.write('\n')
    if args.timer:
        timer = time() - start
        err("Completed in %0.4f seconds (%d hashes per second)\n" %
            (timer, tries[0] / timer))