"""Small native Tk components for the existing Vinyl Studio; no asset dependencies."""
import json
import math
import tkinter as tk
from pathlib import Path

BG='#11100F'


def blend(a,b,t):
    aa=[int(a[i:i+2],16) for i in (1,3,5)]
    bb=[int(b[i:i+2],16) for i in (1,3,5)]
    return '#'+''.join(f'{round(x+(y-x)*t):02x}' for x,y in zip(aa,bb))


def load_preferences(path):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data,dict): return {}
        return {k:v for k,v in data.items() if k in ('settings_expanded','reduced_motion') and type(v) is bool}
    except (OSError,ValueError): return {}


def save_preferences(path,expanded,reduced_motion):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps({'settings_expanded':bool(expanded),'reduced_motion':bool(reduced_motion)}),encoding='utf-8')
    temp.replace(path)


def vinyl_angle(position):
    """One gentle visual revolution per 24 audio seconds (2.5 visual RPM).

    Musical tempo stays in the actual events/playhead; visual speed does not alter audio.
    Pause, seek and restart follow the same absolute audio position without integration drift.
    """
    return float(position)*15.0


def landscape(canvas,width,height,phase=0,background=BG,tag='landscape'):
    """Deterministic layered ink contours; ambient phase is independent of music."""
    canvas.delete(tag)
    w=max(1,width);h=max(1,height)
    # A moon softened at its rim, distant ridges and low-contrast fog.
    cx=w*.74+math.sin(phase)*2;cy=h*.32;radius=min(h*.22,w*.18)
    for i in range(10,0,-1):
        r=radius*(1+i*.025)
        canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill=blend(background,'#AAA197',.13+(10-i)*.035),outline='',tags=tag)
    for layer,(base,amplitude,col) in enumerate([(.52,.15,'#262421'),(.65,.19,'#302D29'),(.80,.18,'#3A3630'),(.97,.16,'#201E1B')]):
        points=[]
        for i in range(81):
            x=w*i/80
            y=h*(base-amplitude*(math.sin(i*.075+layer*.9)+.38*math.sin(i*.19+layer)))
            points.extend((x,y))
        canvas.create_polygon(0,h,*points,w,h,fill=col,outline='',smooth=True,tags=tag)
        canvas.create_line(*points,fill=blend(col,'#AAA197',.13),width=1,smooth=True,tags=tag)
        if layer==1:
            for j in range(12):
                y=h*(.60+j*.008)+math.sin(phase+layer)*1.2
                canvas.create_line(0,y,w,y,fill=blend(col,'#77716B',.10*math.sin(math.pi*j/12)),tags=tag)
    # Sparse deterministic fine ink contours, not music markings.
    for j in range(5):
        points=[]
        for i in range(41):
            points.extend((w*i/40,h*(.87+j*.018-.065*math.sin(i*.12+1))))
        canvas.create_line(*points,fill=blend('#201E1B','#77716B',.08),smooth=True,tags=tag)
    canvas.tag_lower(tag)


class CollapsibleSettings(tk.Frame):
    """Clip an existing widget subtree; collapsing never destroys its variables."""
    def __init__(self,parent,expanded=True,on_change=None,reduced_motion=lambda:False):
        super().__init__(parent,bg='#191817')
        self.expanded=expanded;self.on_change=on_change;self.reduced_motion=reduced_motion
        self._after=None
        self.header=tk.Button(self,text='',command=self.toggle,anchor='w',bg='#211F1D',fg='#F0EAE0',
            activebackground='#393531',activeforeground='#F5F1E9',relief='flat',bd=0,
            font=('Segoe UI',10),padx=9,pady=9)
        self.header.pack(fill='x')
        self.viewport=tk.Frame(self,bg='#191817',height=1)
        self.viewport.pack(fill='x');self.viewport.pack_propagate(False)
        self.content=tk.Frame(self.viewport,bg='#191817',padx=2,pady=8)
        self.content.place(x=0,y=0,relwidth=1)
        self.content.bind('<Configure>',self._content_changed)
        self._set_title()

    def _set_title(self):
        self.header.configure(text=('−' if self.expanded else '+')+'  Generation Settings')

    def _content_changed(self,event=None):
        if self._after is None:
            self.viewport.configure(height=self.content.winfo_reqheight() if self.expanded else 1)

    def toggle(self):
        self.expanded=not self.expanded;self._set_title()
        if self._after is not None:self.after_cancel(self._after);self._after=None
        target=self.content.winfo_reqheight() if self.expanded else 1
        start=self.viewport.winfo_height()
        if self.reduced_motion(): self.viewport.configure(height=target)
        else:
            def frame(step=1):
                t=step/10;ease=t*t*(3-2*t)
                self.viewport.configure(height=round(start+(target-start)*ease))
                self._after=self.after(16,lambda:frame(step+1)) if step<10 else None
            frame()
        if self.on_change:self.on_change()
