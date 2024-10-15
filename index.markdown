---
layout: default
---

<ul class="posts">
  {% for post in site.posts %}
  {% assign post_date = post.date | date: '%m-%d' %}

  <p style="margin-top: 3px; margin-bottom: 3px;">

    {% if post.type == 'external' %}
    <span style="display: inline-block; vertical-align: top">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
        height="1em" width="1em" style="margin-top: 2.5px">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
      </svg>
    </span>
    {% elsif post.type == 'slides' %}
    <span style="display: inline-block; vertical-align: top">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
        height="1em" width="1em" style="margin-top: 2.5px">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5m.75-9l3-3 2.148 2.148A12.061 12.061 0 0116.5 7.605" />
      </svg>
    </span>
    {% elsif post.type == 'code' %}
    <span style="display: inline-block; vertical-align: top">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
        height="1em" width="1em" style="margin-top: 2.5px">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
      </svg>
    </span>
    {% elsif post.type == 'video' %}
    <span style="display: inline-block; vertical-align: top">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
        height="1em" width="1em" style="margin-top: 2.5px">
        <path stroke-linecap="round"
          d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
      </svg>
    </span>
    {% elsif post.type == 'video' %}
    <span>
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
        height="1em" width="1em">
        <path stroke-linecap="round"
          d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
      </svg>
    </span>
    {% else %}
    <span style="display: inline-block; vertical-align: top">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
        height="1em" width="1em" style="margin-top: 2.5px">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
      </svg>
    </span>
    {% endif %}


    {% if post.link %}
    <a href="{{ post.link }}">{{ post.title }}</a> -
    {% else %}
    <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a> -
    {% endif %}

    {% if post_date == "07-01" %}
    <i>{{ post.date | date: "%-d %b %Y" }} 🇨🇦</i>
    {% else %}
    <i>{{ post.date | date: "%-d %b %Y" }}</i>
    {% endif %}
  </p>

  {% endfor %}

</ul>