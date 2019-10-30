from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import pykka


class gui():
    def __init__(self):
        self.gameRoot = Tk()
        self.loggerRoot = Tk()
        self.plot_frame = Frame(self.loggerRoot)
        self.plot_frame.pack()
        self.label = Label(self.gameRoot,text="deep sea treasure hunter")
        self.label.pack()


    def draw_plot(self,data, name, tag_x, tag_y):
        f = Figure(figsize=(5, 5), dpi=100)


        a = f.add_subplot(111)

        a.set_ylabel(tag_y)
        a.set_xlabel(tag_x)
        a.plot([i for i in range(len(data))], data)

        canvas = FigureCanvasTkAgg(f, self.plot_frame)
        canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
        canvas.draw()
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)


    def buffer_frame(self):
        self.buffer_frame = Frame(self.loggerRoot)
        self.buffer_frame.pack(side=BOTTOM)
        button = Button(self.buffer_frame , text="display buffer" ,fg="red" )
        button.pack()

    def create_grid(self,map):
        c = Canvas(self.gameRoot, height=550, width=550, bg='white')
        for i,row in enumerate(map):
            for j,col in enumerate(row):
                if col == -5:
                    color = "blue"
                    col = "AGENT"
                elif col > 0:
                    color = "green"
                else:
                    color = "yellow"
                c.create_rectangle(j*50, i*50, j*50+50, i*50+50, fill=color)
                c.create_text(j*50+25,  i*50+25, text=str(col))
        c.pack(fill=BOTH, expand=True)


    def render(self):
        self.gameRoot.mainloop()
        #self.loggerRoot.mainloop()




def test():
    map = [[  "AG" ,   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
     [   1.,    0.  ,  0. ,   0. ,   0. ,   0. ,   0.,    0.,    0. ,   0.],
     [   0.  ,  2.  ,  0.  ,  0.   , 0.  ,  0.  ,  0.   , 0.  ,  0.  ,  0.],
     [   0.  ,  0.  ,  3.  ,  0.   , 0.  ,  0. ,   0. ,   0.  ,  0. ,   0.],
     [   0.  ,  0.  ,  0.  ,  5.   , 8.  , 16. ,   0.   , 0. ,   0. ,   0.],
     [   0.  ,  0.  ,  0.   , 0.   , 0.  ,  0.  ,  0.  ,  0.  ,  0. ,   0.],
     [   0.  ,  0.  ,  0.  ,  0.   , 0.  ,  0.  , 24. ,  50. ,   0. ,   0.],
     [   0. ,   0.   , 0. ,   0. ,   0.  ,  0.  ,  0.   , 0. ,   0.  ,  0.],
     [   0.  ,  0.  ,  0.  ,  0. ,   0.   , 0. ,   0.  ,  0. ,  74.  ,  0.],
     [   0. ,   0.  ,  0.  ,  0. ,   0.  ,  0.  ,  0.   , 0. ,   0.  ,124.]]
    test_data = [3,4,3,6,7,  0.  ,  0.,5
                 ,5
                 ,6]
    ui = gui()
    ui.draw_plot(test_data,"rewards ", "cumulative reward" , "episode")
    ui.buffer_frame()
    ui.create_grid(map)
    ui.render()


