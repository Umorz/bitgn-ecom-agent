# -*- coding: utf-8 -*-
"""Несколько планов диспетчера подряд: python run_seq.py plan1.json plan2.json ...
   Следующий план стартует, даже если предыдущий упал."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
for plan in sys.argv[1:]:
    print(f"[{time.strftime('%H:%M:%S')}] >>> {plan}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "dispatcher.py"), os.path.join(HERE, plan)], cwd=HERE)
print(f"[{time.strftime('%H:%M:%S')}] все планы пройдены", flush=True)
