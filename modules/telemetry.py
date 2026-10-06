"""Opt-in ORT placement profiling, stored only in isolated state."""
from pathlib import Path
import os
import json
from collections import Counter
import time

ROOT=Path(__file__).resolve().parents[1]
SESSIONS=[]

def profile_options(options,label):
    if os.environ.get('FACEART_PROFILE')=='1':
        folder=ROOT/'state'/'outputs'/'candidate2'/'profiles'
        folder.mkdir(parents=True,exist_ok=True)
        options.enable_profiling=True
        options.profile_file_prefix=str(folder/(label+'-'+str(time.time_ns())))
    return options

def register(session,label):
    if os.environ.get('FACEART_PROFILE')=='1':
        SESSIONS.append((label,session))
    return session

def finish_profiles():
    summaries=[]
    for label,session in SESSIONS:
        try:
            path=session.end_profiling()
            events=json.loads(Path(path).read_text())
            counts,durations=Counter(),Counter()
            for event in events:
                provider=event.get('args',{}).get('provider')
                if event.get('cat')=='Node' and provider:
                    counts[provider]+=1
                    durations[provider]+=event.get('dur',0)
            summaries.append({'session':label,'registered_providers':session.get_providers(),'node_event_counts':dict(counts),'node_duration_sum_ms':{key:value/1000 for key,value in durations.items()},'profile':path})
        except Exception as error:
            summaries.append({'session':label,'error':str(error)})
    SESSIONS.clear()
    return summaries
