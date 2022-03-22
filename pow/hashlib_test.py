from hashlib import sha1 as sha


def main():
    m = sha()
    m.update(b"Nobody inspects")
    m.update(b" the spammish repetition")
    digest1 = m.digest()
    print('digest1: ', digest1)
    print('digest1 size (The size of the resulting hash in bytes.): ',
          m.digest_size)
    print(
        "digest1 block size (The internal block size of the hash algorithm in "
        "bytes.): ",
        m.block_size)
    hexdigest1 = m.hexdigest()
    print('hexdigest1: ', hexdigest1)
    print('length of hexdigest1: ', len(hexdigest1))


if __name__ == "__main__":
    main()