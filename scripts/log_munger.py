from os import path
import re
import sys
import astropy.units as u

def main(filename):
    filename = path.abspath(path.expanduser(filename))

    lines = []
    with open(filename, "r") as f:
        lines = lines + f.readlines()

    pattr = re.compile("saved samples \(([0-9\.]+) seconds\)")
    pattr2 = re.compile("done with star (.*):")
    pattr3 = re.compile("([0-9]+) stars left to process")

    apogee_ids = []
    times = []

    for line in lines:
        try:
            times.append(float(pattr.search(line).groups()[0]))
        except:
            pass

        try:
            apogee_ids.append(pattr2.search(line).groups()[0])
        except:
            pass

        try:
            n_batch = int(pattr3.search(line).groups()[0])
        except:
            pass

    avg_time = sum(times) / len(times) * u.second
    n_left = n_batch - len(apogee_ids)
    time_left = n_left * avg_time

    print('{0} left to process'.format(n_left))
    print('average time per source: {0:.2f}'.format(avg_time))
    print('estimated time left for processing:: {0:.0f}'.format(time_left.to(u.hour)))


if __name__ == '__main__':
    main(sys.argv[0])
