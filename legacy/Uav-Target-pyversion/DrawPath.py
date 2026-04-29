import matplotlib.pyplot as plt
import numpy as np
    
def ACO_DrawPath(route = None,City = None,route1 = None,City1 = None): 
    ## Draw Path Function
# Input
# route     Path to be drawn
# City      Coordinates of each city
    
    figure
    hold('on')
    box('on')
    # xlim([min(City(:,1)-0.01),max(City(:,1)+0.01)]) # Manually set the x-axis range (xlimit).
# ylim([min(City(:,2)-0.01),max(City(:,2)+0.01)]) # Manually set the y-axis range.
    
    # Draw the distribution center point.
    plt.plot(0,0,'bp','MarkerFaceColor','r','MarkerSize',15)
    text(0 + 0.01,0,'飞�器基地')
    # Draw the demand points.
    plt.plot(City(np.arange(1,end()+1),1),City(np.arange(1,end()+1),2),'o','color',np.array([0.5,0.5,0.5]),'MarkerFaceColor','g')
    # Add point number.
    m = 0
    for i in np.arange(1,City.shape[1-1]+1).reshape(-1):
        if City(i,1) != 0 and City(i,2) != 0:
            m = City(i,3)
            text(City(i,1) + 0.002,City(i,2) - 0.002,num2str(m))
    
    plt.axis('equal')
    # Draw the arrow.
#A = City(route+1,:);
    A = City1
    arrcolor = np.random.rand(1,3)
    
    for i in np.arange(2,len(A)+1).reshape(-1):
        arrowx,arrowy = ACO_dsxy2figxy(gca,A(np.arange(i - 1,i+1),1),A(np.arange(i - 1,i+1),2))
        annotation('textarrow',arrowx,arrowy,'HeadLength',8,'HeadWidth',8,'LineWidth',2,'color',arrcolor)
        # Change the color of the next vehicle's route.
        if route1(i) == 0:
            arrcolor = np.random.rand(1,3)
    
    set(gca,'LineWidth',1)
    hold('off')
    plt.xlabel('X�')
    plt.ylabel('Y�')
    plt.title('打击顺序')