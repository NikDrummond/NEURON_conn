# plotting...

import plotly.graph_objects as go

def radius_plot(df,n_mesh, radius = 'radius'):
    """
    Plot nodes within a neuron with mesh, to check the radius sizes
    
    Parameters
    ----------
    
    df:       pandas.DataFrame
             data frame with x,y,z, and radius columns   
    
    n_mesh:  Navis.MeshNeuron object - to be changed
    
    radius:  str
             String of the column name to be used to set the radius value of nodes. default "radius"
    
    
    Returns
    -------
    
    plotly graph object
    
    
    """
    
    data = [go.Scatter3d(
                    x=df['x'],
                    y=df['y'],
                    z=df['z'],
                    mode='markers',
                    marker=dict(size=list(df[radius].values), symbol="circle", opacity = 1,
                                line_width = 0,sizeref = 0.01),
                    hoverinfo = 'none'
                    
            ),
            go.Mesh3d(
                    x = n_mesh.vertices[:,0],
                    y = n_mesh.vertices[:,1],
                    z = n_mesh.vertices[:,2],
                    i = n_mesh.faces[:,0],
                    j = n_mesh.faces[:,1],
                    k = n_mesh.faces[:,2],
                    opacity = 0.4,
                    hoverinfo = 'none')
        ]
    fig=go.Figure(data=data)
    return fig