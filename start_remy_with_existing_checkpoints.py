#!/usr/bin/env python3
import sys
import re
import subprocess
import time

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

# PATH = "/stored_checkpoints"

# checkpoints=stored_checkpoints/1_1_2018-01-24-13:16:43_1 learning_rate=-4.5 reward_type=no_cutoff src/remy-unicorn cf=config/1_1_really_small_buffer_kind_of_slow.cfg
all_file_names = list(map(lambda line: line[:-1], sys.stdin.readlines()))
regexp = re.compile("^(\d+)_(\d+)_(.+)_(\d+)$")

if len(sys.argv) < 4:
	eprint("Ahhh!!! Didn't get all the arguments I wanted! Using default parameters which are probably wrong...")

sloss = sys.argv[1] if len(sys.argv) >= 2 else "kind_of_slow"
learning_rate = sys.argv[2] if len(sys.argv) >= 3 else "-4.5"
reward_type = sys.argv[3] if len(sys.argv) >= 4 else "no_cutoff"

for file_name in all_file_names:
	match = regexp.search(file_name)
	# if acked_match is not None and int(acked_match.group(2)) >= 1:
	assert(match is not None)

	my_command="checkpoints=stored_checkpoints/"+match.group(1)+"_"+match.group(2)+"_"+match.group(3)+"_"+match.group(4)+" learning_rate="+learning_rate+" reward_type="+reward_type+" nohup src/remy-unicorn cf=config/"+match.group(2)+"_"+match.group(1)+"_really_small_buffer_"+sloss+".cfg"

	time.sleep(5)
	print("my_command", my_command)
	subprocess.Popen(["bash", "-c", my_command])
	# subprocess.run(["bash", "-c", my_command])
