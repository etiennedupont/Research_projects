import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

import DDPG_pendulum

DDPG_pendulum.main()

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
