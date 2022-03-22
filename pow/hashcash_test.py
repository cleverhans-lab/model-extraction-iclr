from pow.hashcash import mint_iteractive, generate_challenge, check, _to_binary
from time import time

DAY1 = 60 * 60 * 24  # Seconds in a day


def main():
    num = 16
    print('num: ', num)
    num = hex(num)
    print('hex: ', num)
    bin = _to_binary(num)
    print('bin: ', bin)

    # all_bits = [x * 4 for x in range(1, 26)]
    all_bits = [x for x in range(1, 113)]
    for bits in all_bits:
        start = time()

        # server
        xtype = 'bin'  # 'bin' or 'hex'
        resource = 'model-extraction-warning'
        challenge = generate_challenge(resource=resource, bits=bits)

        # client
        stamp = mint_iteractive(challenge=challenge, bits=bits, xtype=xtype)

        # server
        is_correct = check(stamp=stamp, resource=resource, bits=bits,
                           check_expiration=DAY1, xtype=xtype)

        end = time()
        elapsed_time = end - start
        print('is_correct,', is_correct,
              ',bits,', bits,
              ',elapsed_time,', elapsed_time)


if __name__ == "__main__":
    main()