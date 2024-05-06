$(document).ready(function(){

    var $header = $('header');
    var $body = $('body');
    var $window = $(window);

    // scrolling variables
	var scrolling = false,
    previousTop = 0,
    currentTop = 0,
    scrollDelta = 10,
    scrollOffset = 150;

    // scroll event to show / hide header
    $window.on('scroll', function(){
		if( !scrolling ) {
			scrolling = true;
			(!window.requestAnimationFrame) ? setTimeout(autoHideHeader, 250) : requestAnimationFrame(autoHideHeader);
		}
    });
    
    function autoHideHeader() {
		var currentTop = $window.scrollTop();
        var isNavOpen = $body.hasClass('nav-in');
        if (!isNavOpen) {
            if (previousTop - currentTop > scrollDelta) {
                // if scrolling up...
                $header.removeClass('up');
            } else if( currentTop - previousTop > scrollDelta && currentTop > scrollOffset) {
                // if scrolling down...
                $header.addClass('up');
            }
        }
	   	previousTop = currentTop;
		scrolling = false;
	}

    // toggle sidebar
    $(document).on('click', '.site-nav-toggle, .site-nav a', function() {
        $body.toggleClass('nav-in');
    });

    // replace anchor scroll to offset the fixed header on mobile
    $("a[href^='#']").on('click', function(e) {
        // prevent default anchor click behavior
        e.preventDefault();
     
        var width = $window.width();
        var headerHeight = $header.outerHeight();
        var mobileMaxWidth = 991;
        var offset = 0;

        if (width <= mobileMaxWidth) {
            offset = headerHeight + 10;
        }

        // animate scroll
        $('html, body').animate({
            scrollTop: $(this.hash).offset().top - offset
          }, 200, function(){
        });
    });

    // wrap tables so we can make responsive
    $("table.docutils:not(.field-list,.footnote,.citation)")
        .wrap("<div class='scroll-x'></div>");

});