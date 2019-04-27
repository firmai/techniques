

```python
from bokeh.io import show, output_notebook
from bokeh.palettes import PuBu4
from bokeh.plotting import figure
from bokeh.models import Label
```


```python
output_notebook()
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="a390c47f-10e0-4e58-9dfc-bd71d2e560ba">Loading BokehJS ...</span>
    </div>





```python
# Load in the data
data= [("John Smith", 105, 120),
       ("Jane Jones", 99, 110),
       ("Fred Flintstone", 109, 125),
       ("Barney Rubble", 135, 123),
       ("Mr T", 45, 105)]

limits = [0, 20, 60, 100, 160]
labels = ["Poor", "OK", "Good", "Excellent"]
cats = [x[0] for x in data]
```


```python
# Create the base figure
p=figure(title="Sales Rep Performance", plot_height=350, plot_width=800, y_range=cats)
p.x_range.range_padding = 0
p.grid.grid_line_color = None
p.xaxis[0].ticker.num_minor_ticks = 0
```


```python
# Here's the format of the data we need
print(list(zip(limits[:-1], limits[1:], PuBu4[::-1])))
```

    [(0, 20, '#f1eef6'), (20, 60, '#bdc9e1'), (60, 100, '#74a9cf'), (100, 160, '#0570b0')]



```python
for left, right, color in zip(limits[:-1], limits[1:], PuBu4[::-1]):
    p.hbar(y=cats, left=left, right=right, height=0.8, color=color)
```


```python
show(p)
```



<div class="bk-root">
    <div class="bk-plotdiv" id="e912efec-2f32-48be-ab47-688c96f8f8c3"></div>
</div>





```python
# Now add the black bars for the actual performance
perf = [x[1] for x in data]
p.hbar(y=cats, left=0, right=perf, height=0.3, color="black")
```




<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.models.renderers.GlyphRenderer">GlyphRenderer</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'e93f5798-f4ae-446b-9f3c-942c68bace00', <span id="d67dd6aa-6554-4120-82ae-aa40e0b15eab" style="cursor: pointer;">&hellip;)</span></div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">data_source&nbsp;=&nbsp;ColumnDataSource(id='085e412b-1aaf-43c8-b6e0-139f49f5ffcd', ...),</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">glyph&nbsp;=&nbsp;HBar(id='b92bf553-b73d-48df-a5d1-94c943c7e233', ...),</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hover_glyph&nbsp;=&nbsp;None,</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_event_callbacks&nbsp;=&nbsp;{},</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_property_callbacks&nbsp;=&nbsp;{},</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">level&nbsp;=&nbsp;'glyph',</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">muted&nbsp;=&nbsp;False,</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">muted_glyph&nbsp;=&nbsp;None,</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">nonselection_glyph&nbsp;=&nbsp;HBar(id='927bb98f-15be-42aa-8e52-62b3154cc5a7', ...),</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">selection_glyph&nbsp;=&nbsp;None,</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">subscribed_events&nbsp;=&nbsp;[],</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">view&nbsp;=&nbsp;CDSView(id='729fdab5-46e1-493f-971e-2aa7d7432c5f', ...),</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range_name&nbsp;=&nbsp;'default',</div></div><div class="b4f46ab0-2a50-471a-9c4e-838f581cb27a" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range_name&nbsp;=&nbsp;'default')</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("d67dd6aa-6554-4120-82ae-aa40e0b15eab");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("b4f46ab0-2a50-471a-9c4e-838f581cb27a");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>





```python
show(p)
```



<div class="bk-root">
    <div class="bk-plotdiv" id="0159ae65-dfd1-4490-86fc-2f1cfb53211d"></div>
</div>





```python
# Add the segment for the target
comp = [x[2]for x in data]
p.segment(x0=comp, y0=[(x, -0.5) for x in cats], x1=comp, 
          y1=[(x, 0.5) for x in cats], color="white", line_width=2)
```




<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.models.renderers.GlyphRenderer">GlyphRenderer</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'ee3ab49b-2855-47a8-96eb-8ebeab7b488b', <span id="3d0c87ee-af6f-4f4e-9cdb-5d13bd1d6f8c" style="cursor: pointer;">&hellip;)</span></div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">data_source&nbsp;=&nbsp;ColumnDataSource(id='3d262b3e-9023-44d7-8a83-19c96af775a2', ...),</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">glyph&nbsp;=&nbsp;Segment(id='b354e977-9496-4c93-89a0-e0a2b94e6bbe', ...),</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hover_glyph&nbsp;=&nbsp;None,</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_event_callbacks&nbsp;=&nbsp;{},</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_property_callbacks&nbsp;=&nbsp;{},</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">level&nbsp;=&nbsp;'glyph',</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">muted&nbsp;=&nbsp;False,</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">muted_glyph&nbsp;=&nbsp;None,</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">nonselection_glyph&nbsp;=&nbsp;Segment(id='730690a3-2f7b-4955-b98b-a0144e684e08', ...),</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">selection_glyph&nbsp;=&nbsp;None,</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">subscribed_events&nbsp;=&nbsp;[],</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">view&nbsp;=&nbsp;CDSView(id='1aa6760e-9d05-4593-a619-12c85424c44f', ...),</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range_name&nbsp;=&nbsp;'default',</div></div><div class="ce026221-f668-4d29-b3ec-39cd83489fba" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range_name&nbsp;=&nbsp;'default')</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("3d0c87ee-af6f-4f4e-9cdb-5d13bd1d6f8c");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("ce026221-f668-4d29-b3ec-39cd83489fba");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>





```python
show(p)
```



<div class="bk-root">
    <div class="bk-plotdiv" id="24483e63-6f39-4a77-9fd8-491f565fa805"></div>
</div>





```python
# Add the labels
for start, label in zip(limits[:-1], labels):
    p.add_layout(Label(x=start, y=0, text=label, text_font_size="10pt",
                       text_color='black', y_offset=5, x_offset=15))
```


```python
show(p)
```



<div class="bk-root">
    <div class="bk-plotdiv" id="14ed9770-d624-4ebf-a5de-d06fdd72911b"></div>
</div>



